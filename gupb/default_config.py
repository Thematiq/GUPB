from gupb.controller import keyboard
from gupb.controller import random
from gupb.controller.cynamonka.cynamonka import CynamonkaController
from gupb.controller.bob.agent2 import QBot
from gupb.model.arenas import ArenaDescription
from gupb.scripts import arena_generator

# cynamonka_controller = CynamonkaController("CynamonkaController")
bob = QBot()

CONFIGURATION = {
    'arenas': ['ordinary_chaos'],
    'controllers': [
        # cynamonka_controller,
        bob,
        random.RandomController("Alice"),
        random.RandomController("Bob"),
        random.RandomController("Cecilia"),
        random.RandomController("Darius"),
        random.RandomController("Asd"),
        random.RandomController("Bo"),
        random.RandomController("Ce"),
    ],
    'start_balancing': False,
    'visualise': False,
    'show_sight': bob,
    'runs_no': 100,
    'profiling_metrics': [],
}