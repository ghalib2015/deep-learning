import numpy as np
from skimage import io
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']


class Balls_CF_Seq(Dataset):
    def __init__(self, dir, seq_count, train=0):
        self.dir = dir
        self.seq_count = seq_count
        self.train = train

    # The access is _NOT_ shuffled. The Dataloader will need
    # to do this.
    def __getitem__(self, index):
        # Load bounding boxes and split it up
        if self.train:
            index += 700
        b = np.load("%s/seq_bb_%05d.npy" % (self.dir, index))
        b = torch.from_numpy(b)
        b = list(torch.split(b[b.sum(dim=2) != 0], 3))
        c = torch.empty(20, 3, 4)
        for i in range(20):
            c[i] = b[i]
        x1 = c[:19, 0, :]
        x2 = c[:19, 1, :]
        x3 = c[:19, 2, :]
        return x1, x2, x3, c[19]

    # Return the dataset size
    def __len__(self):
        return self.seq_count


if __name__ == "__main__":
    train_dataset = Balls_CF_Seq("mini_balls_seq", 6299, train=1)
    valid_dataset = Balls_CF_Seq("mini_balls_seq", 699)
    train = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=50, shuffle=True)
    valid = torch.utils.data.DataLoader(valid_dataset,
                                        batch_size=50, shuffle=True)
    print(len(train))
    print(len(valid))

    x1, x2, x3, t = train_dataset.__getitem__(42)
    print(t)
    x1, x2, x3, t = valid_dataset.__getitem__(42)
    print(t)