from .. import Controller
from typing import Any
from gupb.model.characters import ChampionKnowledge, Tabard, ChampionDescription, Facing, Action
from gupb.model.arenas import ArenaDescription
from gupb.model.tiles import TileDescription
import numpy as np
from datetime import datetime
from numpy import ndarray
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

POSSIBLE_RANDOM_ACTIONS = [
    Action.TURN_LEFT,
    Action.TURN_RIGHT,
    Action.STEP_FORWARD,
    Action.ATTACK,
]


class ReplayBuffer(object):
    def __init__(self, mem_size, state_shape, champ_size, n_actions):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=float)
        self.self_state_memory = np.zeros((self.mem_size, champ_size), dtype=float)
        self._state_memory = np.zeros((self.mem_size, *state_shape), dtype=float)
        self._self_state_memory = np.zeros((self.mem_size, champ_size), dtype=float)
        self.action_memory = np.zeros(self.mem_size, dtype=int)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, self_state, action, reward, _state, _self_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.self_state_memory[index] = self_state
        self._state_memory[index] = _state
        self._self_state_memory[index] = _self_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    @staticmethod
    def _softmax(x: ndarray) -> ndarray:
        z = x - max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator / denominator

        return softmax

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        #sotmax na nagrody
        probs = self._softmax(np.abs(self.reward_memory[:max_mem]))

        batch = np.random.choice(max_mem, batch_size, p=probs)

        states = self.state_memory[batch]
        self_states = self.self_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        _states = self._state_memory[batch]
        _self_states = self._self_state_memory[batch]
        done = self.terminal_memory[batch]

        return states, self_states, actions, rewards, _states, _self_states, done


def conv_block(in_channels, out_channels, kernel):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding='same'),
        nn.ReLU(inplace=True),
    )


class Model(nn.Module):
    def __init__(self, alpha, state_shape, self_state_shape, n_actions, name, fname='tmp/model'):
        super(Model, self).__init__()

        self.name = name
        self.save_dir = os.path.join(fname, name)

        ##dane mapy
        self.conv1 = conv_block(state_shape[0], 16, 3)
        self.conv2 = conv_block(16, 32, 3)
        self.conv3 = conv_block(32, 64, 3)
        self.conv4 = conv_block(64, 32, 1)
        ##dane championa
        self.champ_fc1 = nn.Linear(self_state_shape, 128)
        self.champ_fc2 = nn.Linear(128, 256)
        self.champ_fc3 = nn.Linear(256, 256)
        self.champ_fc4 = nn.Linear(256, 256)
        self.champ_fc5 = nn.Linear(256, 256)
        self.champ_fc6 = nn.Linear(256, 256)

        self.fc1 = nn.Linear(3456, 128)
        self.fc2 = nn.Linear(128, 256)
        self.q = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0') if T.cuda.is_available() else T.device('cpu')

        self.loss = nn.MSELoss()

        self.to(self.device)

    def forward(self, state, champ_state):
        state = self.conv1(state)
        state = self.conv2(state)
        state = self.conv3(state)
        state = self.conv4(state)
        state = T.flatten(state, 1)

        champ_state = F.relu(self.champ_fc1(champ_state))
        champ_state = F.relu(self.champ_fc2(champ_state))
        champ_state = F.relu(self.champ_fc3(champ_state))
        champ_state = F.relu(self.champ_fc4(champ_state))
        champ_state = F.relu(self.champ_fc5(champ_state))
        champ_state = F.relu(self.champ_fc6(champ_state))

        state = T.cat([state, champ_state], dim=1)

        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        state = self.q(state)

        return state

    def save_model(self, save_dir=None, filename=None):
        if save_dir is None:
            save_dir = self.save_dir

        if filename is None:
            filename = self.save_file_name
        T.save(self.state_dict(), os.path.join(save_dir, filename))

    def load_model(self, load_dir=None):
        if load_dir is None:
            load_dir = self.save_dir
        self.load_state_dict(T.load(load_dir))


class QBot(Controller):
    _HEALTH_MULTI = 2
    _TILES = {
        'land': 1,
        'sea': 2,
        'wall': 3,
        'menhir': 4,
    }

    _WEAPONS = {
        'knife': 5,
        'bow_loaded': 6,
        'bow_unloaded': 7,
        'sword': 8,
        'axe': 9,
        'amulet': 10
    }

    _EFFECTS = {
        'mist': 11,
        'weaponCut': 12,
    }

    _FACING = {
        Facing.UP: 13,
        Facing.DOWN: 14,
        Facing.LEFT: 15,
        Facing.RIGHT: 16
    }

    _CONSUMABLE = 17

    _HEALTH = 18

    _CHARACTERS = 0

    _N_CHANNELS = 19

    MAPS = {
        'archipelago': (_N_CHANNELS, 50, 50),
        'dungeon': (_N_CHANNELS, 50, 50),
        'fisher_island': (_N_CHANNELS, 50, 50),
        'island': (_N_CHANNELS, 100, 100),
        'isolated_shrine': (_N_CHANNELS, 19, 19),
        'lone_sanctum': (_N_CHANNELS, 19, 19),
        'mini': (_N_CHANNELS, 10, 10),
        'wasteland': (_N_CHANNELS, 50, 50),
    }

    POSSIBLE_ACTIONS = [
        Action.TURN_LEFT,
        Action.TURN_RIGHT,
        Action.STEP_FORWARD,
        Action.ATTACK,
    ]

    BIN2DEC = 2**np.arange(_WEAPONS.__len__())

    SAVE_EVERY_N = 15_000

    def __init__(self):

        self.champ_data: ChampionDescription = None
        self.map_size = (0, 0)
        self.my_data_size = self._N_CHANNELS
        self.transition_cache: dict[str, Any] = {
            'state': None,
            'self_state': None,
            'action': None,
            'reward': 0,
            '_state': None,
            '_self_state': None,
            'done': False
        }

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

    def reset(self, arena_description: ArenaDescription) -> None:
        self.map_size = self.MAPS[arena_description.name]
        if not self.memory:
            self.memory = ReplayBuffer(10_000, self.map_size, self.my_data_size, self.action_size)
        if not self.q:
            self.q = Model(0.001, self.map_size, self.my_data_size, self.action_size, 'q_learning')
            #tu można na double
            self.q.float()
            #todo: to do inita przenieść
            #self.q.load_model('M:\\Studia\\UCZENIE\\BOB\\15-01')

        self.transition_cache: dict[str, Any] = {
            'state': np.zeros(self.map_size),
            'self_state': np.zeros(self._N_CHANNELS),
            'action': 0,
            'reward': 0,
            '_state': np.zeros(self.map_size),
            '_self_state': np.zeros(self._N_CHANNELS),
            'done': False
        }

    def learn(self, batch_size=16):
        if self.memory.mem_cntr < batch_size:
            return

        states, my_states, actions, rewards, _states, _my_states, done = self.memory.sample_buffer(batch_size)

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

            q_target[batch_index, actions] = rewards + self.gamma * q_next[batch_index, max_actions] * done
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
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.min_eps else self.min_eps

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

    def do_reward(self) -> float:

        weapon_change_idx = self.transition_cache['_self_state'][self._WEAPONS['knife']: self._WEAPONS['amulet']+1].dot(self.BIN2DEC) - self.transition_cache['self_state'][self._WEAPONS['knife']: self._WEAPONS['amulet']+1].dot(self.BIN2DEC)
        #todo: reward za hita, negatywna nagroda za mgłę
        #
        #zabicie, miejsce- softmax na zbiernie itemów
        #zapuisywać akcję i wartośćjaką sieć wypluła dla akcji -> sftmax stanów nie po nagrodzie
        # Go! - D/D/D Divine Zero King Rage
        #exploracja?
        vis_ties = np.sum(self.transition_cache['_self_state'][1:5]) - np.sum(self.transition_cache['self_state'][1:5])
        health_diff = self.transition_cache['_self_state'][self._HEALTH] - self.transition_cache['self_state'][self._HEALTH]
        potion = self.transition_cache['_self_state'][self._CONSUMABLE]
        DDD = 0
        return weapon_change_idx + vis_ties + health_diff + potion * self._HEALTH_MULTI

    def serialize_knowledge(self, knowledge: ChampionKnowledge):
        map = np.zeros(self.map_size)

        my_data: TileDescription = knowledge.visible_tiles.pop(knowledge.position)

        temp = np.zeros(self.my_data_size)
        temp[0] = knowledge.position.x
        temp[1] = knowledge.position.y
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
        self.transition_cache['_state'] = map
        self.transition_cache['_self_state'] = my_data

        reward = self.do_reward()
        self.transition_cache['reward'] = reward

        self.memory.store_transition(**self.transition_cache)

        self.learn()

        action_idx = self.choose_action(map, my_data)

        self.transition_cache['state'] = map
        self.transition_cache['self_state'] = my_data
        self.transition_cache['action'] = action_idx

        return self.POSSIBLE_ACTIONS[action_idx]

    def praise(self, score: int) -> None:
        self.transition_cache['reward'] = (3 - score) * 100
        self.transition_cache['done'] = True
        self.memory.store_transition(**self.transition_cache)


    @property
    def name(self) -> str:
        return 'Bob'

    @property
    def preferred_tabard(self) -> Tabard:
        return Tabard.GREEN
