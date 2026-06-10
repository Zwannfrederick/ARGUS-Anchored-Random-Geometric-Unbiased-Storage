from typing import Any, List, Dict
from argus_cache.core.logger import argus_log

class ElasticCacheBalloonDriver:
    """
    KVM/VMware Memory Ballooning concept adapted for KV Cache.
    Dynamically inflates/deflates VRAM constraints based on system-wide memory pressure
    or multi-tenant rebalancing.
    """
    def __init__(self):
        # Maps cache instance -> original max_active_pages (int)
        self.original_max_active_pages: Dict[Any, int] = {}
        # Maps cache instance -> Dict[tier_name, original_max_pages]
        self.original_max_pages: Dict[Any, Dict[str, int]] = {}
        # Maps cache instance -> original max_heat (float)
        self.original_max_heat: Dict[Any, float] = {}
        # Maps cache instance -> current inflation count
        self.inflated_active_pages: Dict[Any, int] = {}

    def _register_cache(self, cache):
        """Registers original configuration of a cache instance if not already registered."""
        if cache not in self.original_max_active_pages:
            self.original_max_active_pages[cache] = cache.max_active_pages
            self.inflated_active_pages[cache] = 0
            
            tier_limits = {}
            for spec in cache.tier_specs:
                tier_limits[spec.name] = spec.max_pages
            self.original_max_pages[cache] = tier_limits
            
            if cache.eviction_policy is not None and hasattr(cache.eviction_policy, 'max_heat'):
                self.original_max_heat[cache] = cache.eviction_policy.max_heat

    def inflate(self, cache):
        """
        Inflates the balloon: dynamically constrains the VRAM limits of the cache
        and forces aggressive page compression / demotion to reclaim memory.
        """
        self._register_cache(cache)
        
        # 1. Reduce active pages limit (minimum 1 active page)
        if cache.max_active_pages > 1:
            cache.max_active_pages -= 1
            self.inflated_active_pages[cache] += 1
            argus_log("WARNING", f"Elastic Cache Balloon INFLATED: max_active_pages reduced from {cache.max_active_pages + 1} to {cache.max_active_pages}", line_no=40)
        else:
            argus_log("INFO", f"Elastic Cache Balloon already fully inflated for active pages (limit={cache.max_active_pages})", line_no=42)

        # 2. Reduce tier limits by 1 (minimum 1 page per tier, except archives which are unlimited with -1)
        for spec in cache.tier_specs:
            if spec.max_pages > 1:
                spec.max_pages -= 1
                argus_log("WARNING", f"Elastic Cache Balloon INFLATED: tier '{spec.name}' max_pages reduced to {spec.max_pages}", line_no=48)

        # 3. Reduce clock sweep max heat and cool down pages if ClockSweepPolicy is used
        if cache.eviction_policy is not None and hasattr(cache.eviction_policy, 'max_heat'):
            # Lower max_heat
            current_max_heat = cache.eviction_policy.max_heat
            cache.eviction_policy.max_heat = max(0.5, current_max_heat * 0.5)
            argus_log("WARNING", f"Elastic Cache Balloon INFLATED: Clock sweep max_heat reduced from {current_max_heat:.2f} to {cache.eviction_policy.max_heat:.2f}", line_no=55)
            
            # Cool down active pages
            for page in cache.active_pages:
                page["referenced"] = 0
                if "heat_register" in page:
                    page["heat_register"] = max(0.0, page["heat_register"] * 0.5)
            
            # Cool down other tiers
            for spec in cache.tier_specs:
                pages_list = cache.pages_by_tier.get(spec.name, [])
                for page in pages_list:
                    page["referenced"] = 0
                    if "heat_register" in page:
                        page["heat_register"] = max(0.0, page["heat_register"] * 0.5)

        # 4. Proactively run the cascade loop to enforce the new reduced limits immediately
        # Demote active pages if count exceeds the new limit
        while len(cache.active_pages) > cache.max_active_pages:
            if cache.eviction_policy is not None:
                demoted_page = cache.eviction_policy.select_victim(cache.active_pages, context={})
                cache.active_pages.remove(demoted_page)
            else:
                cache.active_pages.sort(key=lambda x: x.get('importance_score', 0.0))
                demoted_page = cache.active_pages.pop(0)
            cache._demote_to_next_tier(demoted_page, -1)

        # Demote tier pages if they exceed the new max_pages limits
        for spec in cache.tier_specs:
            pages_list = cache.pages_by_tier.get(spec.name, [])
            while spec.max_pages > 0 and len(pages_list) > spec.max_pages:
                if cache.eviction_policy is not None:
                    victim = cache.eviction_policy.select_victim(pages_list, context={})
                    pages_list.remove(victim)
                else:
                    pages_list.sort(key=lambda p: p.get('importance_score', 0.0))
                    victim = pages_list.pop(0)
                tier_idx = cache.tier_specs.index(spec)
                cache._demote_to_next_tier(victim, tier_idx)

    def deflate(self, cache):
        """
        Deflates the balloon: restores original VRAM limits and configurations
        when memory pressure is relieved.
        """
        self._register_cache(cache)
        
        if self.inflated_active_pages.get(cache, 0) == 0:
            # Not inflated, nothing to restore
            return

        # Restore max_active_pages
        old_active = cache.max_active_pages
        cache.max_active_pages = self.original_max_active_pages[cache]
        self.inflated_active_pages[cache] = 0
        argus_log("INFO", f"Elastic Cache Balloon DEFLATED: max_active_pages restored from {old_active} to {cache.max_active_pages}", line_no=105)

        # Restore tier limits
        tier_limits = self.original_max_pages[cache]
        for spec in cache.tier_specs:
            if spec.name in tier_limits:
                spec.max_pages = tier_limits[spec.name]
                argus_log("INFO", f"Elastic Cache Balloon DEFLATED: tier '{spec.name}' max_pages restored to {spec.max_pages}", line_no=111)

        # Restore clock sweep max heat
        if cache.eviction_policy is not None and hasattr(cache.eviction_policy, 'max_heat'):
            if cache in self.original_max_heat:
                old_heat = cache.eviction_policy.max_heat
                cache.eviction_policy.max_heat = self.original_max_heat[cache]
                argus_log("INFO", f"Elastic Cache Balloon DEFLATED: Clock sweep max_heat restored from {old_heat:.2f} to {cache.eviction_policy.max_heat:.2f}", line_no=118)

    def rebalance(self, caches: List[Any]):
        """
        Rebalances VRAM capacity allocation across multiple cache instances
        based on active attention demands.
        """
        if len(caches) < 2:
            return
            
        # Register all caches first
        for c in caches:
            self._register_cache(c)

        # Calculate activity score for each cache based on page importance
        scores = []
        for c in caches:
            page_importance = sum(p.get("importance_score", 0.0) for p in c.active_pages)
            scores.append((c, page_importance))

        # Sort caches by activity (lowest activity first)
        scores.sort(key=lambda x: x[1])

        # Median split to decide who to inflate vs deflate
        median_idx = len(scores) // 2
        for i, (c, score) in enumerate(scores):
            if i < median_idx:
                # Low activity -> Inflate balloon to release VRAM to the pool
                self.inflate(c)
            else:
                # High activity -> Deflate balloon to reclaim VRAM space
                self.deflate(c)
