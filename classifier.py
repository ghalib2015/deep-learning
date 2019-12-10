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


class SeqClassifier(torch.nn.Module):
    def __init__(self):
        super(SeqClassifier, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.lstm1 = torch.nn.LSTM(4, 100, batch_first=True)
        self.lstm2 = torch.nn.LSTM(4, 100, batch_first=True)
        self.lstm3 = torch.nn.LSTM(4, 100, batch_first=True)
        self.fc = torch.nn.Linear(300, 3 * 4)

        # for training change (1, 1, 100) to (1, 50, 100)
        self.cell1 = (torch.zeros(1, 1, 100).to(device),
                             torch.zeros(1, 1, 100).to(device))
        self.cell2 = (torch.zeros(1, 1, 100).to(device),
                             torch.zeros(1, 1, 100).to(device))
        self.cell3 = (torch.zeros(1, 1, 100).to(device),
                             torch.zeros(1, 1, 100).to(device))

    def forward(self, x1, x2, x3):
        l1, self.cell1 = self.lstm1(x1, self.cell1)
        l1 = l1[:,-1,:]
        l2, self.cell2 = self.lstm2(x2, self.cell2)
        l2 = l2[:, -1, :]
        l3, self.cell3 = self.lstm3(x3, self.cell3)
        l3 = l3[:, -1, :]
        l = torch.cat([l1, l2, l3], dim=1)
        x = self.fc(l)
        return F.relu(x)

 #Another version of the sequence model with faster convergence (faster training)
class NewSeqClassifier(torch.nn.Module):
    def __init__(self):
        super(NewSeqClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(4, 100, batch_first=True)
        self.fc = torch.nn.Linear(100, 100)        
        self.fc2 = torch.nn.Linear(100, 4)        
    def forward(self, x1, x2, x3):                        
        def _forward(xi) :
            l,_ = self.lstm(xi)
            l = l[:,-1,:]
            l = F.relu(self.fc(l))
            l = F.relu(self.fc2(l))
            return l.unsqueeze(1)
        x = torch.cat((_forward(x1),_forward(x2),_forward(x3)), dim=1)        
        return x
