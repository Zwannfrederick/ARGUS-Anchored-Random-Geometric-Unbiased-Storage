"""Static pool shapes must come from a tier's declared bit width.

_allocate_pool_for_tier previously branched on the tier's *name*, so a plugin
tier got no pool at all -- the same name-coupling the C++ codec refactor
removed, still present on the Python side.
"""

import torch

from argus_cache.core.pool_allocator import StaticPoolAllocator
from argus_cache.core.tier_registry import TierSpec

PAGE, BATCH, HEADS, HEAD_DIM = 64, 1, 2, 16


def _allocate(allocator, spec, max_pages=2):
    return allocator.allocate_for_tier(
        spec, max_pages, torch.device("cpu"), torch.float16, BATCH, HEADS, HEAD_DIM
    )


def test_eight_bit_tier_gets_full_length_pool():
    allocator = StaticPoolAllocator(page_size=PAGE)
    pools = _allocate(allocator, TierSpec(name="int8", backend="int8"))

    assert pools["int8_key_q"].shape == (2, BATCH, HEADS, PAGE, HEAD_DIM)
    assert pools["int8_key_q"].dtype == torch.int8


def test_one_bit_tier_pool_is_packed_eight_to_one():
    allocator = StaticPoolAllocator(page_size=PAGE)
    pools = _allocate(allocator, TierSpec(name="one_bit", backend="one_bit"))

    assert pools["one_bit_key_q"].shape[3] == PAGE // 8
    assert pools["one_bit_key_q"].dtype == torch.uint8


def test_affine_tiers_get_min_val_pools_and_sign_packed_tiers_do_not():
    """min_vals is the zero-point of an affine codec. A sign-packed tier has
    no zero-point, and allocating one would waste pool memory per page."""
    allocator = StaticPoolAllocator(page_size=PAGE)

    affine = _allocate(allocator, TierSpec(name="int4", backend="int4"))
    packed = _allocate(allocator, TierSpec(name="one_bit", backend="one_bit"))

    assert "int4_key_min_vals" in affine
    assert "one_bit_key_min_vals" not in packed


def test_plugin_tier_gets_a_pool_sized_from_its_capabilities():
    """The regression: a custom tier used to fall through every name branch."""
    from argus_cache.backends.quantization import INT2Backend
    from argus_cache.plugins import (
        BackendCapabilities,
        NativeCodecSpec,
        register_quantizer,
        unregister_quantizer,
    )

    register_quantizer(
        "plugin_two_bit",
        INT2Backend,
        BackendCapabilities(
            name="plugin_two_bit",
            effective_bits=2.0,
            native_codec=NativeCodecSpec(kind="unsigned_affine", bits=2),
        ),
        replace=True,
    )
    try:
        allocator = StaticPoolAllocator(page_size=PAGE)
        pools = _allocate(
            allocator, TierSpec(name="plugin_two_bit", backend="plugin_two_bit")
        )

        assert pools, "plugin tier received no static pool"
        assert pools["plugin_two_bit_key_q"].shape[3] == PAGE // 4
    finally:
        unregister_quantizer("plugin_two_bit")


def test_projection_tier_gets_no_static_pool():
    """A projection tier's compressed shape depends on the projection rank,
    not on a fixed bit width, so it cannot use a preallocated pool."""
    allocator = StaticPoolAllocator(page_size=PAGE)

    pools = _allocate(allocator, TierSpec(name="jl", backend="jl"))

    assert pools == {}


def test_release_frees_a_tiers_pools():
    allocator = StaticPoolAllocator(page_size=PAGE)
    _allocate(allocator, TierSpec(name="int8", backend="int8"))

    allocator.release("int8")

    assert allocator.get("int8", "key_q") is None


def test_reset_keeps_the_pools_mapping_identity():
    """Callers hold a reference to .pools; reset must clear it in place or
    they would keep reading a stale dict after reallocation."""
    allocator = StaticPoolAllocator(page_size=PAGE)
    handle = allocator.pools
    _allocate(allocator, TierSpec(name="int8", backend="int8"))

    allocator.reset()

    assert allocator.pools is handle
    assert handle == {}
