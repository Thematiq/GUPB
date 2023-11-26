import numpy as np


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

    def store_transition(
        self, state, self_state, action, reward, _state, _self_state, done
    ):
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
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator / denominator

        return softmax

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        # sotmax na nagrody
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
