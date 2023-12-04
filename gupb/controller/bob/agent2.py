import os

from .. import Controller
from .replay_buffer import ReplayBuffer
from .model import Model, CNNPooler
from typing import Any
from gupb.model.characters import (
    ChampionKnowledge,
    Tabard,
    ChampionDescription,
    Facing,
    Action,
)
from gupb.model.arenas import ArenaDescription
from gupb.model.tiles import TileDescription
import numpy as np
import torch as T
import wandb
from collections import defaultdict
from datetime import datetime

POSSIBLE_RANDOM_ACTIONS = [
    Action.TURN_LEFT,
    Action.TURN_RIGHT,
    Action.STEP_FORWARD,
    Action.ATTACK,
]


class QBot(Controller):
    _HEALTH_MULTI = 2
    _TILES = {
        "land": 1,
        "sea": 2,
        "wall": 3,
        "menhir": 4,
    }

    _PLAYER_X = 0
    _PLAYER_Y = 1

    _WEAPONS = {
        "knife": 5,
        "bow_loaded": 6,
        "bow_unloaded": 7,
        "sword": 8,
        "axe": 9,
        "amulet": 10,
    }

    _EFFECTS = {
        "mist": 11,
        "weaponCut": 12,
    }

    _FACING = {Facing.UP: 13, Facing.DOWN: 14, Facing.LEFT: 15, Facing.RIGHT: 16}

    _CONSUMABLE = 17

    _HEALTH = 18

    _CHARACTERS = 0

    _N_CHANNELS = 19

    MAPS = {
        "archipelago": (_N_CHANNELS, 50, 50),
        "dungeon": (_N_CHANNELS, 50, 50),
        "fisher_island": (_N_CHANNELS, 50, 50),
        "island": (_N_CHANNELS, 100, 100),
        "isolated_shrine": (_N_CHANNELS, 19, 19),
        "lone_sanctum": (_N_CHANNELS, 19, 19),
        "mini": (_N_CHANNELS, 10, 10),
        "wasteland": (_N_CHANNELS, 50, 50),
        "ordinary_chaos": (_N_CHANNELS, 24, 24),
    }

    POSSIBLE_ACTIONS = [
        Action.TURN_LEFT,
        Action.TURN_RIGHT,
        Action.STEP_FORWARD,
        Action.ATTACK,
        Action.STEP_BACKWARD,
        Action.STEP_LEFT,
        Action.STEP_RIGHT,
    ]

    BIN2DEC = 2 ** np.arange(_WEAPONS.__len__())

    SAVE_EVERY_N = 5_000

    def __init__(self):
        self._model_cls = CNNPooler
        self._steps_survived = 0
        self.champ_data: ChampionDescription = None
        self.map_size = (0, 0)
        self.my_data_size = self._N_CHANNELS
        self.transition_cache: dict[str, Any] = {
            "state": None,
            "self_state": None,
            "action": None,
            "reward": 0,
            "_state": None,
            "_self_state": None,
            "done": False,
        }

        self.run = self.__init__wandb()
        self._log_scores = defaultdict(lambda: [])

        self.action_size = 4

        self.memory = None
        self.q = None
        self.epsilon = 1
        self.eps_dec = 0.0001
        self.min_eps = 0.005
        self.gamma = 0.99

    def __eq__(self, other) -> bool:
        return isinstance(other, QBot)

    def __hash__(self) -> int:
        return -1

    def __init__wandb(self):
        return wandb.init(
            project="gupb-model",
            name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    def __log_to_wandb(self):
        log_data = {}
        scores = [sum(x) for x in zip(*self._log_scores.values())]
        log_data["avg_reward"] = np.mean(scores)
        log_data["std_reward"] = np.std(scores)
        log_data["steps_survived"] = self._steps_survived
        log_data |= {k: wandb.Histogram(v) for k, v in self._log_scores.items()}
        self.run.log(log_data)

    def reset(self, game_no, arena_description: ArenaDescription) -> None:
        self.map_size = self.MAPS[arena_description.name]
        if not self.memory:
            self.memory = ReplayBuffer(10_000, self.map_size, self.my_data_size, self.action_size)
        if not self.q:
            self.q = Model(0.001, self.map_size, self.my_data_size, self.action_size, 'q_learning')
            if path := get_most_recent_file() is not None:

                self.q.load_model(get_most_recent_file())

            self.q.float()

        if len(self._log_scores) > 0:
            self.__log_to_wandb()

        self._steps_survived = 0
        self._log_scores = defaultdict(lambda: [])

        self.transition_cache: dict[str, Any] = {
            "state": np.zeros(self.map_size),
            "self_state": np.zeros(self._N_CHANNELS),
            "action": 0,
            "reward": 0,
            "_state": np.zeros(self.map_size),
            "_self_state": np.zeros(self._N_CHANNELS),
            "done": False,
        }

    def learn(self, batch_size=16):
        if self.memory.mem_cntr < batch_size:
            return

        (
            states,
            my_states,
            actions,
            rewards,
            _states,
            _my_states,
            done,
        ) = self.memory.sample_buffer(batch_size)

        states = T.tensor(states, dtype=T.float32).to(self.q.device)
        _states = T.tensor(_states, dtype=T.float32).to(self.q.device)
        variables = T.tensor(my_states, dtype=T.float32).to(self.q.device)
        _variables = T.tensor(_my_states, dtype=T.float32).to(self.q.device)

        q_pred = self.q(states, variables)
        with T.no_grad():
            q_next = self.q.forward(_states, _variables).cpu().detach().numpy()
            q_target = q_pred.cpu().detach().numpy().copy()

            max_actions = np.argmax(q_next, axis=1)

            batch_index = np.arange(batch_size, dtype=np.int32)

            q_target[batch_index, actions] = (
                    rewards + self.gamma * q_next[batch_index, max_actions] * done
            )
            q_target = T.tensor(q_target).to(self.q.device)

        self.q.optimizer.zero_grad()
        loss = self.q.loss(q_pred, q_target).to(self.q.device)
        loss.backward()
        self.q.optimizer.step()
        self.update_epsilon_value()
        if self.memory.mem_cntr % self.SAVE_EVERY_N == 0:
            filename = datetime.now().strftime("%H-%M")
            self.q.save_model(filename=filename)

    def update_epsilon_value(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.min_eps else self.min_eps
        )

    def choose_action(self, state, my_data):
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            state = np.expand_dims(state, 0)
            my_data = np.expand_dims(my_data, 0)
            state = T.tensor(state, dtype=T.float32).to(self.q.device)
            variables = T.tensor(my_data, dtype=T.float32).to(self.q.device)
            actions = self.q.forward(state, variables)
            action = T.argmax(actions).item()

        return action

    def __calculate_mist_reward(self):
        _STAYED_MIST_PENALTY = -5
        _ENTERED_MIST_PENALTY = -20
        _EXITED_MIST_REWARD = 10

        current_state = self.transition_cache["_state"]
        current_champ = self.transition_cache["_self_state"]
        current_mist_info = current_state[
            self._EFFECTS["mist"],
            int(current_champ[self._PLAYER_X]),
            int(current_champ[self._PLAYER_Y]),
        ]

        prev_state = self.transition_cache["state"]
        prev_champ = self.transition_cache["self_state"]
        prev_mist_info = prev_state[
            self._EFFECTS["mist"],
            int(prev_champ[self._PLAYER_X]),
            int(prev_champ[self._PLAYER_Y]),
        ]

        reward = 0

        if current_mist_info == 1:
            reward += _STAYED_MIST_PENALTY
        if current_mist_info == 0 and prev_mist_info == 1:
            reward += _EXITED_MIST_REWARD
        elif current_mist_info == 1 and prev_mist_info == 0:
            reward += _ENTERED_MIST_PENALTY

        return reward

    def do_reward(self) -> float:
        weapon_change_idx = self.transition_cache["_self_state"][
                            self._WEAPONS["knife"]: self._WEAPONS["amulet"] + 1
                            ].dot(self.BIN2DEC) - self.transition_cache["self_state"][
                                                  self._WEAPONS["knife"]: self._WEAPONS["amulet"] + 1
                                                  ].dot(
            self.BIN2DEC
        )
        # todo: reward za hita, negatywna nagroda za mgłę
        #
        # zabicie, miejsce- softmax na zbiernie itemów
        # zapuisywać akcję i wartośćjaką sieć wypluła dla akcji -> sftmax stanów nie po nagrodzie
        # Go! - D/D/D Divine Zero King Rage
        # exploracja?
        vis_ties = np.sum(self.transition_cache["_self_state"][1:5]) - np.sum(
            self.transition_cache["self_state"][1:5]
        )
        health_diff = (
                self.transition_cache["_self_state"][self._HEALTH]
                - self.transition_cache["self_state"][self._HEALTH]
        )
        potion = self.transition_cache["_self_state"][self._CONSUMABLE]
        mist = self.__calculate_mist_reward()

        for k, v in zip(
                ["visible_tiles", "hp_diff", "potions", "mist"],
                [vis_ties, health_diff, potion, mist],
        ):
            self._log_scores[k].append(v)

        DDD = 0
        return (
                weapon_change_idx
                + vis_ties
                + health_diff
                + potion * self._HEALTH_MULTI
                + mist
        )

    def serialize_knowledge(self, knowledge: ChampionKnowledge):
        map = np.zeros(self.map_size)

        my_data: TileDescription = knowledge.visible_tiles.pop(knowledge.position)

        temp = np.zeros(self.my_data_size)
        temp[self._PLAYER_X] = knowledge.position.x
        temp[self._PLAYER_Y] = knowledge.position.y
        temp[self._WEAPONS[my_data.character.weapon.name]] = 1
        temp[self._FACING[my_data.character.facing]] = 1
        temp[self._HEALTH] = my_data.character.health
        # coords, values = zip(*knowledge.visible_tiles.items())

        # cord_data = np.empty(self.map_size, dtype=object)
        # cord_data[coords] = values

        for coord, value in knowledge.visible_tiles.items():
            map[self._TILES[value.type], coord[0], coord[1]] = 1
            if value.loot:
                map[self._WEAPONS[value.loot.name], coord[0], coord[1]] = 1

            if value.character:
                map[self._CHARACTERS, coord[0], coord[1]] = 1
                map[self._FACING[value.character.facing], coord[0], coord[1]] = 1
                map[self._WEAPONS[value.character.weapon.name], coord[0], coord[1]] = 1

            if value.consumable:
                map[self._CONSUMABLE, coord[0], coord[1]] = 1

            for eff in value.effects:
                map[self._EFFECTS[eff.type], coord[0], coord[1]] = 1

        return map, temp

    def decide(self, knowledge: ChampionKnowledge) -> Action:
        map, my_data = self.serialize_knowledge(knowledge)
        self.transition_cache["_state"] = map
        self.transition_cache["_self_state"] = my_data

        reward = self.do_reward()
        self.transition_cache["reward"] = reward

        self.memory.store_transition(**self.transition_cache)

        self.learn()

        action_idx = self.choose_action(map, my_data)

        self.transition_cache["state"] = map
        self.transition_cache["self_state"] = my_data
        self.transition_cache["action"] = action_idx
        self._steps_survived += 1

        return self.POSSIBLE_ACTIONS[action_idx]

    def praise(self, score: int) -> None:
        self.transition_cache["reward"] = (3 - score) * 100
        self.transition_cache["done"] = True
        self.memory.store_transition(**self.transition_cache)

    @property
    def name(self) -> str:
        return "Bob"

    @property
    def preferred_tabard(self) -> Tabard:
        return Tabard.GREEN


def get_most_recent_file(directory='\\tmp\\model\\q_learning'):
    par = os.getcwd()
    directory = par + directory
    try:

        if not os.path.exists(directory):
            # Create the directory and its parent directories if they don't exist
            os.makedirs(directory)
            return None
        else:
            # Get a list of all files in the directory
            files = [os.path.join(directory, file) for file in os.listdir(directory) if
                     os.path.isfile(os.path.join(directory, file))]

            if not files:
                print("No files found in the directory.")
                return None

            # Sort files based on modification time (most recent first)
            most_recent_file = max(files, key=os.path.getmtime)

        return most_recent_file
    except OSError as e:
        print(f"Error: {e}")
        return None
