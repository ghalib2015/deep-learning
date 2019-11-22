import torch
import torch.nn.functional as F
from dataset_det import Balls_CF_Detection


class ColorClassifier(torch.nn.Module):
    def __init__(self):
        super(ColorClassifier, self).__init__()
        # convolutional network
        self.conv1 = torch.nn.Conv2d(3, 20, 5, 1, 0)
        self.conv2 = torch.nn.Conv2d(20, 50, 3, 2, 0)
        self.conv3 = torch.nn.Conv2d(50, 20, 5, 1, 0)
        # MLP network
        self.fc1 = torch.nn.Linear(20 * 9 * 9, 300)
        self.fc2 = torch.nn.Linear(300, 9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 20 * 9 * 9)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class BoxClassifier(torch.nn.Module):
    def __init__(self, colorModel):
        super(BoxClassifier, self).__init__()
        # previous model
        self.model = colorModel
        # convolutional network
        self.conv1 = torch.nn.Conv2d(3, 20, 5, 1, 0)
        self.conv2 = torch.nn.Conv2d(20, 50, 3, 2, 0)
        self.conv3 = torch.nn.Conv2d(50, 20, 5, 1, 0)
        # MLP network
        self.fc1 = torch.nn.Linear(20 * 9 * 9, 300)
        self.fc2 = torch.nn.Linear(309, 9 * 4)

    def forward(self, x):
        x1 = self.model(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 20 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, x1), 1)
        return F.relu(self.fc2(x)).view(-1, 9, 4)

