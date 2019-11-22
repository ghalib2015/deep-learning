import torch
import os
from prediction import load, run, performance

def stats(model, dataset):
    print(performance(model))
    for i in range(2100):
        img, p, pose = dataset.__getitem__(i)
        run(model, img, i)

# model, dataset = load("boxClassifier")
# stats(model, dataset)


model, dataset = load("boxClassifier")
for i in range(20):
    img, p, pose = dataset.__getitem__(i)
    run(model, img, i)