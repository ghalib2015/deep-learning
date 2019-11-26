import torch
from classifier import SeqClassifier
from dataset_seq import Balls_CF_Seq


def load(name):
    dataset = Balls_CF_Seq("mini_balls_seq", 699)
    model = SeqClassifier()
    model.cuda()
    model.load_state_dict(torch.load(name + ".pth")["state_dict"])
    model.eval()
    model.train(False)
    return model, dataset


model, dataset = load("seqClassifier")

x1, x2, x3, t = dataset.__getitem__(42)
y = model(x1.view(1, 19, 4).cuda(), x2.view(1, 19, 4).cuda(), x3.view(1, 19, 4).cuda()).detach()
print(y)
print(t)

#
# valid = torch.utils.data.DataLoader(dataset,
#                                     batch_size=50, shuffle=True)
# x1, x2, x3, t = next(iter(valid))
# y = model(x1.cuda(), x2.cuda(), x3.cuda()).detach()
# for i in range(50):
#     print(y[i].view(3, 4))
#     print(t[i])
#     print("_______________________________________")
