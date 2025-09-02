from __future__ import annotations

"""
Scenario and map generation utilities.

Keep scenario builders lightweight and deterministic where practical.
"""

from .blockage import BlockageScenarioConfig, create_blockage_scenario

__all__ = [
    "BlockageScenarioConfig",
    "create_blockage_scenario",
]

