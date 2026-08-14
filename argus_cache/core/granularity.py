"""Variable granularity: splitting and merging KV pages at runtime.

A page is the unit of eviction, so its size is a policy knob: large pages
amortize bookkeeping but force cold and hot tokens to share a fate, while
small pages evict precisely at the cost of more of them. This subsystem lets
a single cache use both -- splitting cold mega-pages into micro-pages and
merging hot runs of micro-pages back.

**Status: experimental, and restricted to ACTIVE (uncompressed) pages.**
Splitting a page that already lives in a compressed tier would round-trip it
through the Python backend, which packs along the sequence axis, while the
native kernels for the same tier names pack along head_dim. The two layouts
are byte-incompatible, so a page split here and later resurrected through the
native path would decode garbage. See ``rebalance`` for the details.

Invariants this subsystem must preserve, whatever the tier:

* **Conservation** -- split then merge returns the original token count, and
  on an uncompressed page returns the original content.
* **Identity** -- every page produced draws a fresh id from the native
  counter. A reused id aliases a live page in every id-keyed structure.
* **Packing compatibility** -- ``micro_page_size`` stays a multiple of 8, or
  the first cascade into a sub-byte tier packs a partial byte.
"""

from __future__ import annotations

import argus_cpp_backend
import torch

from .logger import argus_log
from .outliers import _apply_outlier_restoration, isolate_outliers


class GranularityManager:
    """Splits and merges pages on behalf of a :class:`PagedDynamicKVCache`.

    Holds no state of its own: the pages, tier specs, and JL operators all
    belong to the cache, and this collaborator only encodes the policy for
    reshaping them.
    """

    def __init__(self, cache, micro_page_size: int) -> None:
        self.cache = cache
        self._micro_page_size = micro_page_size

    @property
    def micro_page_size(self) -> int:
        """The cache's live value, not a copy.

        ``micro_page_size`` is writable on the cache (benchmarks and the
        balloon driver retune it), and a stale copy here would split pages at
        one size while the rest of the system assumed another.
        """
        return getattr(self.cache, "micro_page_size", self._micro_page_size)

    def _write_compressed_field(self, page, prefix, comp):
        """
        Writes a pluggable backend.compress() dict ({'q', 'scales', 'min_vals'})
        into the C++ Page struct's fixed fields (<prefix>_compressed/_scale/_min).
        The C++ struct only holds one scalar scale/min per tensor (matching its
        own hardcoded fp8/int8/int4/int2/one_bit tiers), so per-channel scale
        tensors from Python backends are collapsed to their mean. This trades a
        little precision for structural compatibility on the split/merge path.
        """
        page[f'{prefix}_compressed'] = comp['q']
        scales = comp.get('scales')
        min_vals = comp.get('min_vals')
        page[f'{prefix}_scale'] = float(scales.float().mean().item()) if scales is not None else 1.0
        page[f'{prefix}_min'] = float(min_vals.float().mean().item()) if min_vals is not None else 0.0

    def split(self, page, tier_name=None):
        """
        Splits a Mega-page (size page_size) into multiple Micro-pages (size micro_page_size).
        """
        micro_size = self.micro_page_size
        current_size = page.get('page_size', self.cache.page_size)
        if current_size <= micro_size or current_size % micro_size != 0:
            return []

        num_splits = current_size // micro_size
        split_pages = []

        if 'key_compressed' in page or 'value_compressed' in page:
            spec = self.cache.tier_name_to_spec.get(tier_name)
            if spec is None:
                return []

            decomp_args = {"seq_dim": -2}
            if self.cache._is_projection_tier(tier_name):
                recon_op = self.cache.get_jl_reconstruction_operator(
                    page['key_proj'].device, page['key_proj'].dtype,
                    seq_len=page.get('page_size', self.cache.page_size))
                decomp_args['recon_operator'] = recon_op

            k_raw = spec.backend.decompress(page['key_compressed'], **decomp_args)
            v_raw = spec.backend.decompress(page['value_compressed'], **decomp_args)

            k_raw = _apply_outlier_restoration(k_raw, page, key='key')
            v_raw = _apply_outlier_restoration(v_raw, page, key='value')

            for i in range(num_splits):
                start = i * micro_size
                end = start + micro_size
                k_part = k_raw[..., start:end, :]
                v_part = v_raw[..., start:end, :]

                if spec.use_outlier_isolation:
                    k_part_norm, k_part_out, k_part_mask = isolate_outliers(k_part, self.cache.threshold_sigma)
                    v_part_norm, v_part_out, v_part_mask = isolate_outliers(v_part, self.cache.threshold_sigma)
                    part_k_out_indices = torch.nonzero(k_part_mask).to(torch.int16)
                    part_k_out_values = k_part_out[k_part_mask]
                    part_v_out_indices = torch.nonzero(v_part_mask).to(torch.int16)
                    part_v_out_values = v_part_out[v_part_mask]
                else:
                    k_part_norm, v_part_norm = k_part, v_part
                    part_k_out_indices, part_k_out_values = None, None
                    part_v_out_indices, part_v_out_values = None, None

                comp_args = {"seq_dim": -2}
                if spec.is_projection:
                    w_proj = self.cache.get_jl_projection_matrix(k_part_norm.device, k_part_norm.dtype, k_part_norm.shape[-2])
                    comp_args['w_proj'] = w_proj

                part_k_comp = spec.backend.compress(k_part_norm, **comp_args)
                part_v_comp = spec.backend.compress(v_part_norm, **comp_args)

                part_page = argus_cpp_backend.create_page()
                part_page['page_id'] = self.cache._cpp_manager.next_page_id()
                part_page['tier_name'] = tier_name
                part_page['orig_dtype'] = k_part_norm.dtype
                self._write_compressed_field(part_page, 'key', part_k_comp)
                self._write_compressed_field(part_page, 'value', part_v_comp)
                part_page['pool_idx'] = -1
                part_page['attention_sum'] = page.get('attention_sum', 0.0) / num_splits
                part_page['last_step_accessed'] = page.get('last_step_accessed', self.cache.generation_step)
                part_page['importance_score'] = page.get('importance_score', 0.0)
                part_page['page_size'] = micro_size

                # NOTE: Outliers are currently unsupported in C++ Page struct
                # If needed, we must add key_out_indices etc. to C++ Page.

                split_pages.append(part_page)
        else:
            k_raw = page['key']
            v_raw = page['value']

            for i in range(num_splits):
                start = i * micro_size
                end = start + micro_size
                k_part = k_raw[..., start:end, :]
                v_part = v_raw[..., start:end, :]

                part_page = argus_cpp_backend.create_page()
                part_page['page_id'] = self.cache._cpp_manager.next_page_id()
                part_page['tier_name'] = 'active'
                part_page['orig_dtype'] = k_part.dtype
                part_page['key'] = k_part
                part_page['value'] = v_part
                part_page['pool_idx'] = -1
                part_page['attention_sum'] = page.get('attention_sum', 0.0) / num_splits
                part_page['last_step_accessed'] = page.get('last_step_accessed', self.cache.generation_step)
                part_page['importance_score'] = page.get('importance_score', 0.0)
                part_page['page_size'] = micro_size
                split_pages.append(part_page)

        return split_pages

    def merge(self, pages, tier_name=None):
        """
        Merges a list of Micro-pages back into a single Mega-page (size page_size).
        """
        if not pages:
            return None

        total_size = sum(p.get('page_size', self.cache.page_size) for p in pages)
        k_list = []
        v_list = []
        spec = self.cache.tier_name_to_spec.get(tier_name) if tier_name else None

        for p in pages:
            if 'key_compressed' in p or 'value_compressed' in p:
                decomp_args = {"seq_dim": -2}
                if self.cache._is_projection_tier(tier_name):
                    recon_op = self.cache.get_jl_reconstruction_operator(
                        p['key_proj'].device, p['key_proj'].dtype,
                        seq_len=p.get('page_size', self.cache.page_size))
                    decomp_args['recon_operator'] = recon_op

                k_raw = spec.backend.decompress(p['key_compressed'], **decomp_args)
                v_raw = spec.backend.decompress(p['value_compressed'], **decomp_args)

                k_raw = _apply_outlier_restoration(k_raw, p, key='key')
                v_raw = _apply_outlier_restoration(v_raw, p, key='value')
            else:
                k_raw = p['key']
                v_raw = p['value']

            k_list.append(k_raw)
            v_list.append(v_raw)

        k_merged = torch.cat(k_list, dim=-2)
        v_merged = torch.cat(v_list, dim=-2)

        next_id = self.cache._cpp_manager.next_page_id()

        if tier_name is not None and spec is not None:
            if spec.use_outlier_isolation:
                k_norm, k_out, k_mask = isolate_outliers(k_merged, self.cache.threshold_sigma)
                v_norm, v_out, v_mask = isolate_outliers(v_merged, self.cache.threshold_sigma)
                k_out_indices = torch.nonzero(k_mask).to(torch.int16)
                k_out_values = k_out[k_mask]
                v_out_indices = torch.nonzero(v_mask).to(torch.int16)
                v_out_values = v_out[v_mask]
            else:
                k_norm, v_norm = k_merged, v_merged
                k_out_indices, k_out_values = None, None
                v_out_indices, v_out_values = None, None

            comp_args = {"seq_dim": -2}
            if spec.is_projection:
                w_proj = self.cache.get_jl_projection_matrix(k_norm.device, k_norm.dtype, k_norm.shape[-2])
                comp_args['w_proj'] = w_proj

            key_comp = spec.backend.compress(k_norm, **comp_args)
            value_comp = spec.backend.compress(v_norm, **comp_args)

            merged_page = argus_cpp_backend.create_page()
            merged_page['page_id'] = next_id
            merged_page['tier_name'] = tier_name
            merged_page['orig_dtype'] = k_norm.dtype
            self._write_compressed_field(merged_page, 'key', key_comp)
            self._write_compressed_field(merged_page, 'value', value_comp)
            merged_page['pool_idx'] = -1
            merged_page['attention_sum'] = sum(p.get('attention_sum', 0.0) for p in pages)
            merged_page['last_step_accessed'] = max(p.get('last_step_accessed', self.cache.generation_step) for p in pages)
            merged_page['importance_score'] = max(p.get('importance_score', 0.0) for p in pages)
            merged_page['page_size'] = total_size
            # NOTE: outliers skipped for now as C++ doesn't support them
        else:
            merged_page = argus_cpp_backend.create_page()
            merged_page['page_id'] = next_id
            merged_page['tier_name'] = 'active'
            merged_page['orig_dtype'] = k_merged.dtype
            merged_page['key'] = k_merged
            merged_page['value'] = v_merged
            merged_page['pool_idx'] = -1
            merged_page['attention_sum'] = sum(p.get('attention_sum', 0.0) for p in pages)
            merged_page['last_step_accessed'] = max(p.get('last_step_accessed', self.cache.generation_step) for p in pages)
            merged_page['importance_score'] = max(p.get('importance_score', 0.0) for p in pages)
            merged_page['page_size'] = total_size

        return merged_page

    def rebalance(self):
        """
        Scans pages, splits cold Mega-pages into Micro-pages,
        and merges hot contiguous Micro-pages into Mega-pages.
        """
        micro_size = self.micro_page_size

        # 1. Manage active pages
        new_active = []
        i = 0
        while i < len(self.cache.active_pages):
            page = self.cache.active_pages[i]
            p_size = page.get('page_size', self.cache.page_size)

            if p_size == self.cache.page_size and page.get('importance_score', 0.0) < 0.5:
                splits = self.split(page)
                if splits:
                    new_active.extend(splits)
                    argus_log("INFO", f"Splitting Page {page['page_id']} (ACTIVE) -> {len(splits)} Micro-pages", line_no=500)
                    i += 1
                    continue

            needed_pages = self.cache.page_size // micro_size
            if p_size == micro_size and i + needed_pages <= len(self.cache.active_pages):
                candidate_pages = self.cache.active_pages[i : i + needed_pages]
                if all(p.get('page_size', self.cache.page_size) == micro_size and p.get('importance_score', 0.0) > 1.5 for p in candidate_pages):
                    merged = self.merge(candidate_pages)
                    if merged is not None:
                        new_active.append(merged)
                        argus_log("INFO", f"Merging {len(candidate_pages)} Micro-pages -> Page {merged['page_id']} (ACTIVE)", line_no=510)
                        i += needed_pages
                        continue

            new_active.append(page)
            i += 1
        
        # Modify active pages
        self.cache.active_pages = new_active

        # 2. Manage compressed tiers
        #
        # Splitting/merging a page that's already in a compressed tier would
        # decompress and re-compress it via the pluggable Python backend
        # (spec.backend.compress/decompress, called with seq_dim=-2 — it
        # packs along the sequence axis). C++'s own dequant kernels for these
        # same tier names (fp8/int8/int4/int2/one_bit) always pack along the
        # last axis (head_dim) instead, with a hardcoded, unrelated layout.
        # The two are byte-incompatible: a page split here and later
        # resurrected through the C++ path reads back the wrong shape/values.
        # Until the two compression implementations are unified, variable
        # granularity is restricted to ACTIVE (uncompressed FP16) pages,
        # which have no packing format to clash over. Intentionally a no-op
        # below — left structured for when tier-level splitting is revisited.
        for spec in self.cache.tier_specs:
            pages_list = self.cache.pages_by_tier.get(spec.name, [])
            new_list = list(pages_list)

            # Modify pages            # Apply changes
            # Since self.cache.pages_by_tier returns the dictionary directly, we can assign the new list to it
            pages_dict = self.cache.pages_by_tier
            pages_dict[spec.name] = new_list
            self.cache.pages_by_tier = pages_dict
        self.cache._invalidate_decompressed_cache()
