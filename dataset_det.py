import numpy as np
from skimage import io
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']

class Balls_CF_Detection(Dataset):
    def __init__(self, dir, image_count, transform=None, train=0):
        self.transform = transform
        self.dir = dir
        self.image_count = image_count
        self.train = train

    # The access is _NOT_ shuffled. The Dataloader will need
    # to do this.
    def __getitem__(self, index):
        if self.train:
            index = index + 2100
        img = io.imread("%s/img_%05d.jpg"%(self.dir,index))
        img = np.asarray(img)
        img = img.astype(np.float32)
        
        # Dims in: x, y, color
        # should be: color, x, y
        img = np.transpose(img, (2,0,1))
        
        img = torch.tensor(img)
        if self.transform is not None:
            img = self.transform(img)

        # Load presence and bounding boxes and split it up
        p_bb = np.load("%s/p_bb_%05d.npy"%(self.dir,index))
        p  = p_bb[:,0]
        bb = p_bb[:,1:7]
        return img, p, bb

    # Return the dataset size
    def __len__(self):
        return self.image_count
        
if __name__ == "__main__":
    # train_dataset = Balls_CF_Detection ("../mini_balls/train", 20999,
    #     transforms.Normalize([128, 128, 128], [50, 50, 50]))
    train_dataset = Balls_CF_Detection ("train", 20999)

    img,p,b = train_dataset.__getitem__(40)

    print ("Presence:")
    print (p)

    print ("Pose:")
    print (b)