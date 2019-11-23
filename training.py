import torch
from classifier import ColorClassifier
from dataset_det import Balls_CF_Detection
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# environment settings
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(torch.cuda.current_device()))
writer = SummaryWriter("runs/color_classifier.01")

# dataset
train_dataset = Balls_CF_Detection("train", 18899, train=1)
valid_dataset = Balls_CF_Detection("valid", 2099)

# dataloader
training = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=50, shuffle=True)
validation = torch.utils.data.DataLoader(valid_dataset,
                                         batch_size=50, shuffle=True)

# model
model = ColorClassifier().to(device)
model.cuda()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size_1 = 378
batch_size_2 = 42
dataset_1 = 18900 * 9
dataset_2 = 2100 * 9
train_loss = 0
train_error = 0
validation_loss = 0
validation_error = 0

for epoch in range(200):
    print("starting epoch:", epoch, "loss:", "training", train_loss, "validation", validation_loss, "error:",
          "training", train_error, "validation", validation_error)

    train_loss = 0
    train_error = 0
    validation_loss = 0
    validation_error = 0

    for batch, (train_img, train_presence, train_box) in enumerate(training):

        if (batch % 50) == 0:
            print("batch:", batch)

        optimizer.zero_grad()
        out1 = model(train_img.to(device))

        loss = criterion(out1.to(device), train_presence.to(device))
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        out1[out1 > 0.5] = 1
        out1[out1 < 0.5] = 0
        train_error += (out1 != train_presence.to(device)).sum().item()

    for batch, (valid_img, valid_presence, valid_box) in enumerate(validation):

        if (batch % 20) == 0:
            print("batch:", batch)

        out2 = model(valid_img.to(device))

        loss = criterion(out2.to(device), valid_presence.to(device))
        validation_loss += loss.item()

        out2[out2 > 0.5] = 1
        out2[out2 < 0.5] = 0
        validation_error += (out2 != valid_presence.to(device)).sum().item()

    train_loss = train_loss / batch_size_1
    train_error = train_error / dataset_1

    validation_loss = validation_loss / batch_size_2
    validation_error = validation_error / dataset_2

    writer.add_scalars("loss", {"training": train_loss, "validation": validation_loss}, epoch)
    writer.add_scalars("error", {"training": train_error, "validation": validation_error}, epoch)

# save the results
torch.save({"state_dict": model.state_dict(),
            "epoch": 200},
           "colorClassifier.pth")
writer.close()
