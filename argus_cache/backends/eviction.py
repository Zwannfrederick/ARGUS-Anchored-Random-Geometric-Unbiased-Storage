from typing import List, Dict, Any
from argus_cache.core.tier_registry import EvictionPolicy

class ImportanceSortPolicy(EvictionPolicy):
    """Importance-Scored Least Important page replacement policy (v0.1.8 baseline)."""
    def select_victim(self, pages: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        if not pages:
            raise ValueError("No pages to select victim from.")
        # Sort pages by importance_score (ascending) and select the lowest
        sorted_pages = sorted(pages, key=lambda p: p.get("importance_score", 0.0))
        return sorted_pages[0]

    def on_access(self, page: Dict[str, Any], attention_score: float, step: int):
        # Access statistics are tracked on the page itself in memory manager.
        pass


class ClockSweepPolicy(EvictionPolicy):
    """
    PostgreSQL-inspired Clock Sweep page replacement policy with second-chance reference bits 
    and attention heat registers.
    """
    def __init__(self, max_heat: float = 3.0):
        self.hand = 0
        self.max_heat = max_heat

    def select_victim(self, pages: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        if not pages:
            raise ValueError("No pages to select victim from.")

        # Ensure clock hand is within bounds of current pages list
        if self.hand >= len(pages):
            self.hand = 0

        num_pages = len(pages)
        # Scan pages, up to two full clock sweeps
        for _ in range(num_pages * 2):
            page = pages[self.hand]
            
            referenced = page.get("referenced", 0)
            heat = page.get("heat_register", 0.0)

            if referenced > 0:
                # Give a second chance: decrement referenced count/bit
                page["referenced"] = referenced - 1
                # Cool down the heat register
                page["heat_register"] = max(0.0, heat * 0.5)
                # Advance clock hand
                self.hand = (self.hand + 1) % len(pages)
            elif heat > 0.5:
                # Cool down heat register
                page["heat_register"] = max(0.0, heat - 0.5)
                # Advance clock hand
                self.hand = (self.hand + 1) % len(pages)
            else:
                # Found a victim!
                victim = page
                # Advance hand past the victim before returning
                self.hand = (self.hand + 1) % len(pages)
                return victim

        # Fallback: if all pages are hot, evict the current page and advance hand
        victim = pages[self.hand]
        self.hand = (self.hand + 1) % len(pages)
        return victim

    def on_access(self, page: Dict[str, Any], attention_score: float, step: int):
        # Set reference bit to indicate recent access
        page["referenced"] = 1
        # Accumulate attention weight in heat register, capped at max_heat
        current_heat = page.get("heat_register", 0.0)
        page["heat_register"] = min(self.max_heat, current_heat + attention_score)

