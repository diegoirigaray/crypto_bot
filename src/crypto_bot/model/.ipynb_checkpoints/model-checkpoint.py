import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def save_weights(net, net_id, path):
    if not os.path.exists(path):
        os.makedirs(path)
    w_path = "{}/{}.pth".format(path, net_id)
    open(w_path, 'w')
    torch.save(net.state_dict(), w_path)


def load_weights(net, net_id, path):
    w_path = "{}/{}.pth".format(path, net_id)
    try:
        weights = torch.load(w_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(weights)
        return True
    except FileNotFoundError:
        print("**Weights for net_id: {} were not found in path: {}**".format(net_id, path))
        return False


class ConvBotMulti(nn.Module):
    def __init__(self, n_features):
        super(ConvBotMulti, self).__init__()
        # Input timeseries with 'n_features' channels
        self.conv1 = nn.Conv1d(n_features, 128, 3, padding='same')
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, 5, padding='same')
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, 8, padding='same')
        self.bn3 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x


class ConvBotBinary(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # Input timeseries with 'n_features' channels
        self.conv1 = nn.Conv1d(n_features, 32, 3, padding='same')
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, 5, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        
        self.do = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(64, 32, 5, padding='same')
        self.bn3 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.do(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        return x


class ICLayer(nn.Module):
    def __init__(self, n_inputs, p):
        super(ICLayer, self).__init__()
        self.bn = nn.BatchNorm1d(n_inputs)
        self.do = nn.Dropout(p)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.do(x)
        return x


class ConvBotBinaryIC(nn.Module):
    def __init__(self, n_features):
        super(ConvBotBinaryIC, self).__init__()
        # Input timeseries with 'n_features' channels
        self.ic1 = ICLayer(n_features, 0.2)
        self.conv1 = nn.Conv1d(n_features, 32, 3, padding='same')

        self.ic2 = ICLayer(32, 0.2)
        self.conv2 = nn.Conv1d(32, 64, 5, padding='same')

        self.ic3 = ICLayer(64, 0.2)
        self.conv3 = nn.Conv1d(64, 32, 8, padding='same')

        self.ic4 = ICLayer(32, 0.2)
        self.fc1 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(self.ic1(x)))
        x = F.relu(self.conv2(self.ic2(x)))
        x = F.relu(self.conv3(self.ic3(x)))
        x = F.adaptive_avg_pool1d(self.ic4(x), 1)
        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc1(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel_sizes=[8, 5, 3]):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_sizes[0], padding='same')
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_sizes[1], padding='same')
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes, kernel_sizes[2], padding='same')
        self.bn3 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(planes)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, n_features):
        super(ResNet, self).__init__()

        self.block1 = ResBlockIC(n_features, 64)
        self.block2 = ResBlockIC(64, 32)
        self.block3 = ResBlockIC(32, 32)
        self.fc1 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        return x


class ResBlockIC(nn.Module):
    def __init__(self, in_planes, planes, kernel_sizes=[3, 5, 8], ps=[0.2, 0.2, 0.2]):
        super(ResBlockIC, self).__init__()
        self.ic1 = ICLayer(in_planes, ps[0])
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_sizes[0], padding='same')
        self.ic2 = ICLayer(planes, ps[1])
        self.conv2 = nn.Conv1d(planes, planes, kernel_sizes[1], padding='same')
        self.ic3 = ICLayer(planes, ps[2])
        self.conv3 = nn.Conv1d(planes, planes, kernel_sizes[2], padding='same')

        self.shortcut = nn.Sequential()
        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(planes)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(self.ic1(x)))
        out = F.relu(self.conv2(self.ic2(out)))
        out = self.conv3(self.ic3(out))
        out += self.shortcut(x)
        out =  F.relu(out)
        return out


class ResNetIC(nn.Module):
    def __init__(self, n_features):
        super(ResNetIC, self).__init__()

        self.block1 = ResBlockIC(n_features, 64)
        self.block2 = ResBlockIC(64, 32)
        self.block3 = ResBlockIC(32, 32)
        self.fc1 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        return x


class ResNetTest(nn.Module):
    def __init__(self, n_features):
        super(ResNetTest, self).__init__()

        self.block1 = ResBlock(n_features, 32)
        self.do1 = nn.Dropout(0.4)
        # self.block2 = ResBlock(16, 16)
        # self.do2 = nn.Dropout(0.4)
        # self.block3 = ResBlock(64, 32)
        # self.do3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.do1(self.block1(x))
        # x = self.do1(self.block2(x))
        # x = self.do1(self.block3(x))
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        # x = self.fc1(x)
        return x