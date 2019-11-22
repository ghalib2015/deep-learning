# Written by Fabien Baradel

import ipdb
from tqdm import tqdm
import os
import numpy as np
import argparse
import random
from PIL import Image
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from dataset_det import Balls_CF_Detection, COLORS

def show_img(np_array_uint8, out_fn):
    if len(np_array_uint8.shape) == 3:
        img = Image.fromarray(np_array_uint8, 'RGB')
    elif len(np_array_uint8.shape) == 2:
        img = Image.fromarray(np_array_uint8)
    else:
        raise NameError('Unknown data type to show.')

    img.save(out_fn)
    img.show()

def show_bboxes(rgb_array, np_bbox, list_colors, out_fn='./bboxes_on_rgb.png'):
  """ Show the bounding box on a RGB image
  rgb_array: a np.array of shape (H,W,3) - it represents the rgb frame in uint8 type
  np_bbox: np.array of shape (9,4) and a bbox is of type [x1,y1,x2,y2]
  list_colors: list of string of length 9
  """
  assert np_bbox.shape[0] == len(list_colors)
  r=rgb_array.numpy()
  r = np.uint8(r)
  r = np.transpose(r, (1,2,0))

  img_rgb = Image.fromarray(r, 'RGB')
  draw = ImageDraw.Draw(img_rgb)
  N = np_bbox.shape[0]
  for i in range(N):
    color = COLORS[i]
    x_1, y_1, x_2, y_2 = np_bbox[i]
    draw.rectangle(((x_1, y_1), (x_2, y_2)), outline=color, fill=None)

  img_rgb.show()
  img_rgb.save(out_fn)

if __name__ == "__main__":            
      dataset = Balls_CF_Detection("train",20999)

      # Get a single image from the dataset and display it
      img,p,pose = dataset.__getitem__(9)

      print (img.shape)
      print (pose.shape)
      print(pose)

      show_bboxes(img, pose, COLORS, out_fn='_x.png')