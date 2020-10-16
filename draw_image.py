# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 01:57:34 2020

@author: malrawi
"""

# https://www.learnopencv.com/mask-r-cnn-instance-segmentation-with-pytorch/

import numpy as np
from PIL import Image, ImageChops
import torchvision.transforms as T
import random
import cv2
import torch

# from chanel_dataset import get_class_names
# INSTANCE_CATEGORY_NAMES = get_clothCoParse_class_names()
# INSTANCE_CATEGORY_NAMES = get_class_names()

# saving segmented cloths
def save_masks_as_images(img_name, masks, path, file_name, labels):
    img = Image.open(img_name)
    for i in range(len(masks)):        
        image_A = ImageChops.multiply(img, Image.fromarray(255*masks[i]).convert('RGB') )
        image_A.save(path+file_name+labels[i]+'.png')

def get_prediction(model, img, threshold, device, INSTANCE_CATEGORY_NAMES):
      
  img = T.ToTensor()(img).to(device)
  img = img.unsqueeze(dim=0)
  
  with torch.no_grad(): 
      pred = model(img)
  pred_score = list(pred[0]['scores'].cpu().numpy())  
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  masks = (pred[0]['masks']>0.5).squeeze().cpu().numpy()
  pred_class = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return masks, pred_boxes, pred_class

def random_colour_masks(image):
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask


def instance_segmentation_api(model, img, device, class_names, threshold=0.5, rect_th=3, text_size=1, text_th=3):
  
  # img.show()  # before overlaying the masks
  masks, boxes, pred_cls = get_prediction(model, img, threshold, device, class_names)  
  img= np.array(img)
  for i in range(len(masks)):
    rgb_mask = random_colour_masks(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
  
  # img = Image.fromarray(img)
  # img.show()  # after overlaying the masks
  # img.save('C:/Users/msalr/Desktop/test.png', 'png')
  
  return masks, pred_cls

