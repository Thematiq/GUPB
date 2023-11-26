from .agent import FSMBot
from .agent2 import QBot

__all__ = ["FSMBot", "POTENTIAL_CONTROLLERS"]

POTENTIAL_CONTROLLERS = [
    FSMBot()
    # Qbot()
]
