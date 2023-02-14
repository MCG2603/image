# image

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 00:50:20 2021

@author: mirac
"""
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import inspect
import json
import functools
import matplotlib 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image as im 
import pyscreeze
import time
from pyautogui import *
import pyautogui
from scipy import misc
import glob



    

dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())


train_ds7=[]
test_ds7=[]
train_ds, val_ds = random_split(dataset, [50000, 10000])
len(train_ds), len(val_ds)


k=0

for  images,label in train_ds:
    if len(train_ds7)>100:
        break
    if label==7:
          images=images.reshape(1,784)
          train_ds7.append(images)
         
          if len(train_ds7)>100:
              for i in range(784):
                   if i!=0:
                        test_ds7.append(k)
                        k=0
                      
                      
                   for j in train_ds7:
                   
                       k+=float(j[0][i])
                     
                  
test_ds7.append(k)              
print(len(test_ds7)) 

i=0
j=0      
for i in range(len(test_ds7)):
    test_ds7[i]=round(test_ds7[i]/len(test_ds7),3)
train_ds00=[]
diff=np.zeros((10,1))
u=0
k=0
l=0
i=0
j=0
test_ds7= np.array(test_ds7)
test_ds7= test_ds7.reshape(28,28)

for images, label in train_ds:
    
    images=images.reshape(28,28)
    images=images.numpy()
    
    if u>9:
        
        break
    while label==u:
 
      train_ds00.append(images)
      
       
      for i in range(28):
         for j in range(28):
          diff[u]+=(train_ds00[u][i][j]-test_ds7[i][j])*(train_ds00[u][i][j]-test_ds7[i][j])
              
              
              
          
      u+=1   
         
                  
print(diff)                  
test_ds7= np.array(test_ds7)
test_ds7= test_ds7.reshape(28,28)

plt.imshow(test_ds7, cmap='gray')   

images,label=train_ds[49900]

images=images.reshape(28,28)
images=images.numpy()


i=0
train_ds6=[]
diff=[]
diff2=[ 0,  0,  0,  0,  0,  0,  0,  0, 18, 19, 19, 15,  5,  5,  5,  5,  5,  5,
  4,  4,  4,  4,  4,  3,  3,  2,  4,  4,]
diff3=[ 0,  0,  0,  0,  0,  0,  0,  9, 12, 12, 13, 13, 13, 13, 12, 12,  3,  3,
  2,  3,  4,  4,  3,  3,  4,  4,  2,  0,]
diff4=[ 0,  0,  0,  0,  0,  0,  0, 12, 12, 12, 11,  4,  5,  4,  3,  4,  3,  3,
  4,  3,  4,  4,  4,  4,  4,  4,  4,  0,]
diff5=[0, 0, 0, 0, 0, 0, 6, 7, 7, 7, 4, 4, 4, 5, 3, 5, 3, 4, 4, 2, 4, 2, 3, 3,
 2, 2, 0, 0,]
l=0
k=0
m=[]

