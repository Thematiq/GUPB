import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch as T


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

