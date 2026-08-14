# Re-export to eliminate code duplication.
#
# This module previously held a second, divergent implementation of
# PagedDynamicQuantizedCache that had drifted 121 lines from the canonical one
# and silently ignored pipeline= and balloon_driver= -- callers got a cache
# that discarded their tier configuration without raising. Keep this file a
# pure re-export; do not add logic here.
from argus_cache.models.attention_wrapper import *  # noqa: F401,F403
from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache  # noqa: F401
