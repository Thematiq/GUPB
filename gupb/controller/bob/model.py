import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch as T
import torch

from math import floor


def conv_block(in_channels, out_channels, kernel):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding='same'),
        nn.ReLU(inplace=True),
    )


class AbstractModel(nn.Module):
    def __init__(self, name, fname='tmp/model', device=None):
        super().__init__()
        self.name = name
        self.save_dir = os.path.join(fname)
        os.makedirs(self.save_dir, exist_ok=True)

        if device is None:
            self.device = T.device('cuda:0') if T.cuda.is_available() else T.device('cpu')
        else:
            self.device = device

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




class Model(AbstractModel):
    def __init__(self, alpha, state_shape, self_state_shape, n_actions, name, fname='tmp/model'):
        super(Model, self).__init__(name, fname, None)

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


class CNNPooler(AbstractModel):
    def build_conv_block(self, in_channels, out_channels, kernel=2, pool=2):
        mid_channels = (in_channels + out_channels) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel).to(self.device),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel).to(self.device),
            nn.ReLU(),
            nn.MaxPool2d(pool).to(self.device)
        )

    def __init__(self, alpha, state_shape, self_state_shape, n_actions, name, fname='tmp/model2'):
        super().__init__(name, fname, None)

        self._block_1 = self.build_conv_block(state_shape[0], 32)
        self._block_2 = self.build_conv_block(32, 64)
        self._block_3 = self.build_conv_block(64, 256)

        self._state_mlp = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU()
        )

        self._self_state_mlp = nn.Sequential(
            nn.Linear(self_state_shape, out_features=64),
            nn.ReLU(),
            nn.Linear(64, out_features=32),
            nn.ReLU()
        )

        self._final_proc = nn.Sequential(
            nn.Linear(in_features=96, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=n_actions),
            nn.Softmax()
        )

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, state, self_state):
        state = self._block_1(state)
        state = self._block_2(state)
        state = self._block_3(state)
        # Now we got 256 len vector
        gap = torch.mean(state, dim=(2, 3))

        state_res = self._state_mlp(gap)
        self_state_res = self._self_state_mlp(self_state)

        feat = torch.cat([state_res, self_state_res], dim=1)
        return self._final_proc(feat)


