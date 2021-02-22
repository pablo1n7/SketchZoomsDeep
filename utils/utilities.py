from visdom import Visdom
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from scipy import stats
from PIL import Image
import pandas as pd
import numpy as np
import random
import torch
import glob
import cv2
import re
from torch.autograd import Variable
import matplotlib.pyplot as plt

def get_windows(img, i, j, zoom):
    neighbors = []
    w, h,_ = img.shape
    ee_w, ee_h = zoom,zoom
    neighbors = img[i-ee_w:ee_w+i, j-ee_h:ee_h+j]
    return neighbors

def compute_zooms(img, z_zoom=[1.5, 0.5, 0.3], show=False):
    zooms = [] 
    #img = cv2.resize(np.array(img), (512, 512))
    for zind, z in enumerate(z_zoom):
        #print(z)
        img_z = (get_windows(np.array(img), 256, 256, int(100 * z)))
        img_z = cv2.resize(img_z, (224, 224))
        zooms.append(img_z)
    
    if show:
        plt.figure(figsize=(12,5))
        plt.subplot(1, 3, 1)
        plt.imshow(zooms[0])
        plt.plot([112], [112], "ro")
        plt.subplot(1, 3, 2)
        plt.imshow(zooms[1])
        plt.plot([112], [112], "ro")
        plt.subplot(1, 3, 3)
        plt.imshow(zooms[2])
        plt.plot([112], [112], "ro")
        plt.show()
        
    return zooms

import torchvision.transforms as transforms

def compute_feature(img_real, pts=[[256, 256]], t=256, nivel=0, transform_torch=None, z_zoom=[1.8, 1.3, 0.8], show=False ):
    features = [] 
    t_rot = transform_torch.transforms[0]
    t_flip = transform_torch.transforms[1]
    for i, p in enumerate(pts):
        x = p[0] 
        y = p[1]
        t_rot.center=(x, y)
        img = t_rot(t_flip(img_real.crop((x-t, y-t, x+t, y+t))))
        d = np.array(img)
        d[d[:, :, 3] == 0] = [255, 255, 255, 1]
        d = d[:, :, [0, 1, 2]]
        features = features + compute_zooms(d, z_zoom, show)
    features = img_to_tensor(features, transform=transforms.Compose(transform_torch.transforms[2:]))
    
    # normalize -1,1
    #for i in range(features.shape[0]):
    #    minval = features[i,:,:].min()
    #    maxval = features[i,:,:].max()
    #    features[i,:,:] = (2*(features[i,:,:] - minval) / (maxval-minval))-1

    return features


def img_to_tensor(img_points, transform = None):
        imgs_tensor = torch.Tensor()  
        for img in img_points:
            img = np.array(img)
            img = Image.fromarray(normalize(img).astype('uint8'),'RGB')
            if transform is not None:
                img = transform(img)
            imgs_tensor = torch.cat((imgs_tensor, img), 0)
        return imgs_tensor

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))            
    return arr

def array_to_tensor(arrays, transform):
    imgs_tensor = torch.Tensor()  
    for img in arrays:
        img = np.array(img)
        img = Image.fromarray(normalize(img).astype('uint8'),'RGB')
        if transform is not None:
            img = transform(img)
            imgs_tensor = torch.cat((imgs_tensor, img), 0)
    return imgs_tensor

class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main', port=8097):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}
        self.scores_window = None
        self.image_window = None

    def plot(self, var_name, split_name, x, y, x_label='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=x_label,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

    def close_window(self, var_name):
        self.viz.close(self.plots[var_name])
        del self.plots[var_name]
        
    def images(self, images):
        if self.image_window != None:
            self.viz.close(self.image_window)
            
        self.image_window = self.viz.images(images, nrow=3, env=self.env,
                                            opts=dict(nrow=2, title='Images Batch'))
