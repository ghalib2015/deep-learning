import torch
from dataset_det import Balls_CF_Detection, COLORS
from vis import show_bboxes
from classifier import BoxClassifier, ColorClassifier


def load(name):
    dataset = Balls_CF_Detection("valid", 2099)
    colormodel = ColorClassifier()
    colormodel.load_state_dict(torch.load("colorClassifier.pth")["state_dict"])
    colormodel.eval()
    colormodel.train(False)
    model = BoxClassifier(colormodel)
    model.load_state_dict(torch.load(name + ".pth")["state_dict"])
    model.eval()
    model.train(False)
    return model, dataset


def run(model, img, number):
    box = model(img.view(1, 3, 100, 100)).view(9, 4)
    if box[8, :].sum() > 0:
        if box[8, 1] == 0:
            box[8, 1] = box[8, 3] - 11
    show_bboxes(img, box, COLORS, out_fn=str(number) + '.png')


def performance(name):
    validation = 0
    criterion = torch.nn.MSELoss()
    dataset = Balls_CF_Detection("valid", 2099)
    colormodel = ColorClassifier()
    colormodel.load_state_dict(torch.load("colorClassifier.pth")["state_dict"])
    colormodel.eval()
    colormodel.train(False)
    model = BoxClassifier(colormodel)
    model.load_state_dict(torch.load(name + ".pth")["state_dict"])
    model.eval()
    model.train(False)
    for i in range(2100):
        img, p, pose = dataset.__getitem__(i)
        out = model(img.view(1, 3, 100, 100)).view(9, 4)
        if out[8, :].sum() > 0:
            if out[8, 1] == 0:
                out[8, 1] = out[8, 3] - 11
        loss = criterion(out, torch.from_numpy(pose))
        validation += loss.item()
    return validation / 2100
