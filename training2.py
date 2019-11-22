import torch
from classifier import ColorClassifier
from classifier import BoxClassifier
from dataset_det import Balls_CF_Detection
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# environment settings
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(torch.cuda.current_device()))
writer = SummaryWriter("runs/box_classifier.01")

# dataset
train_dataset = Balls_CF_Detection("train", 18899, train=1)
valid_dataset = Balls_CF_Detection("valid", 2099)

# dataloader
training = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=50, shuffle=True)
validation = torch.utils.data.DataLoader(valid_dataset,
                                         batch_size=50, shuffle=True)

# model
colormodel = ColorClassifier().to(device)
colormodel.cuda()
colormodel.load_state_dict(torch.load("colorClassifier.pth")["state_dict"])
colormodel.eval()
colormodel.train(False)
boxModel = BoxClassifier(colormodel).to(device)
boxModel.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(boxModel.parameters(), lr=0.001)

batch_size_1 = 378
batch_size_2 = 42
train_loss = 0
validation_loss = 0

for epoch in range(2000):
    print("starting epoch:", epoch, "loss:", "training", train_loss, "validation", validation_loss)

    train_loss = 0
    validation_loss = 0

    for batch, (train_img, train_presence, train_box) in enumerate(training):

        if (batch % 50) == 0:
            print("batch:", batch)

        optimizer.zero_grad()
        out1 = boxModel(train_img.to(device))

        loss = criterion(out1.to(device), train_box.to(device))
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    for batch, (valid_img, valid_presence, valid_box) in enumerate(validation):

        if (batch % 50) == 0:
            print("batch:", batch)

        out2 = boxModel(valid_img.to(device))

        loss = criterion(out2.to(device), valid_box.to(device))
        validation_loss += loss.item()

    train_loss = train_loss / batch_size_1
    validation_loss = validation_loss / batch_size_2

    writer.add_scalars("loss", {"training": train_loss, "validation": validation_loss}, epoch)

    if (epoch % 100) == 0:
        torch.save({"state_dict": boxModel.state_dict(),
                    "epoch": epoch},
                   str(epoch)+"_intermediate.pth")

# save the results
torch.save({"state_dict": boxModel.state_dict(),
            "epoch": 2000},
           "boxClassifier.pth")
writer.close()
