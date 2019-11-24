import torch
from dataset_seq import Balls_CF_Seq
from classifier import SeqClassifier
from torch.utils.tensorboard import SummaryWriter
from random import random

# environment settings
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(torch.cuda.current_device()))
writer = SummaryWriter("runs/seq_classifier.01")

# dataset
train_dataset = Balls_CF_Seq("mini_balls_seq", 6299, train=1)
valid_dataset = Balls_CF_Seq("mini_balls_seq", 699)

# dataloader
training = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=50, shuffle=True)
validation = torch.utils.data.DataLoader(valid_dataset,
                                         batch_size=50, shuffle=True)
# model
model = SeqClassifier().to(device)
model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size_1 = 126
batch_size_2 = 14
train_loss = 0
validation_loss = 0

for epoch in range(200):
    print("starting epoch:", epoch, "loss:", "training", train_loss, "validation", validation_loss)

    train_loss = 0
    validation_loss = 0

    for batch, (x1, x2, x3, t) in enumerate(training):

        if (batch % 50) == 0:
            print("batch:", batch)

        optimizer.zero_grad()
        y = model(x1.to(device), x2.to(device), x3.to(device)).view(-1, 3, 4)

        loss = criterion(y.to(device), t.to(device))
        train_loss += loss.item()

        loss.backward(retain_graph=True)
        optimizer.step()
        del loss
        model.cell1[0].detach_()
        model.cell1[1].detach_()
        model.cell2[0].detach_()
        model.cell2[1].detach_()
        model.cell3[0].detach_()
        model.cell3[1].detach_()

    if (epoch % 10) == 0:
        torch.save({"state_dict": model.state_dict(),
                    "epoch": epoch},
                   str(epoch)+"_seq.pth")

    for batch, (x1, x2, x3, t) in enumerate(validation):

        if (batch % 5) == 0:
            print("batch:", batch)

        y = model(x1.to(device), x2.to(device), x3.to(device)).view(-1, 3, 4)

        loss = criterion(y.to(device), t.to(device))
        validation_loss += loss.item()

        del loss
        model.cell1[0].detach_()
        model.cell1[1].detach_()
        model.cell2[0].detach_()
        model.cell2[1].detach_()
        model.cell3[0].detach_()
        model.cell3[1].detach_()

    train_loss = train_loss / batch_size_1

    validation_loss = validation_loss / batch_size_2

    writer.add_scalars("loss", {"training": train_loss, "validation": validation_loss}, epoch)

# save the results
torch.save({"state_dict": model.state_dict(),
            "epoch": 200},
           "seqClassifier.pth")
writer.close()
