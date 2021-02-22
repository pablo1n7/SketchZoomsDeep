from torch.utils.data import DataLoader,Dataset
from scipy import stats
from PIL import Image
import pandas as pd
import numpy as np
import random
import torch
import glob
import cv2
import re

from utils.utilities import *

class SketchNetworkDataset(Dataset): 
    
    def __init__(self, corr_files, data_sketch_root, n=1000, n_all=10000, transform=None):
        self.corr_files = corr_files
        self.data_sketch_root = data_sketch_root
        self.transform = transform
        self.n = n
        self.n_all = n_all
        self.prob = stats.rv_discrete(name='custm', values=([0, 1], 
                                                            [0.95, 0.05]))
        self.n_part_len = self.n_all//self.n
        self.part = 0
        self.prob_rotations = stats.rv_discrete(name='custm', 
                                                values=([0, 1, 2, 3], 
                                                        [0.25, 0.25, 0.25, 0.25]))
        self.rotations = [lambda x : x, np.rot90, 
                          lambda x: np.rot90(np.rot90(x)), 
                          lambda x: np.rot90(np.rot90(np.rot90(x)))]
        self.prob_pixear_image = stats.rv_discrete(name='custm', values=([0, 1, 2], [0.60, 0.20, 0.20]))
        self.pixear = [lambda x : x, 
                       lambda x: cv2.resize(x, (64, 64), interpolation=cv2.INTER_CUBIC), 
                       lambda x: cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)]
        self.init_csv()
        self.vista_diferente = 0
        self.vista_igual = 0

    def init_csv(self):
        if self.n_part_len<=0:
            self.part = 0
            self.n_part_len = self.n_all//self.n
    
        self.df = pd.read_csv(self.corr_files, header=0, index_col=0, sep=",", skiprows=self.part, nrows=self.n)
        self.df.reset_index(drop=True, inplace=True)


    def get_point(self, index):
        points_df = self.df.loc[[index]]
        row = np.array(points_df).flatten()
        md1 = row[0]
        md2 = row[1]
        points = np.array(row[2:],dtype=str)
        return md1, md2, points[:-1], points[-1] 
    
    
    def img_to_tensor(self, img_points):
        imgs_tensor = torch.Tensor()  
        for img in img_points:
            img = np.array(img)
            img = Image.fromarray(normalize(img).astype('uint8'),'RGB')
            if self.transform is not None:
                img = self.transform(img)
                imgs_tensor = torch.cat((imgs_tensor, img), 0)
        return imgs_tensor
    
    def zooms(self, img_points):
        image_original = Image.open(img_points)
        image_original = image_original.convert('RGB')
        image_original = np.array(image_original)
        image_original = Image.fromarray(normalize(image_original).astype('uint8'),'RGB')
        image_original = np.array(image_original)
        imgs = []
        for z in [1.5, 1, 0.5]:
            z = z + np.random.uniform(-0.3, 0.3)
            img = get_windows(image_original, 256, 256, int(100 * z))
            img = self.rotations[self.prob_rotations.rvs()](img)
            img = self.pixear[self.prob_pixear_image.rvs()](img)
            imgs.append(img)
        return self.img_to_tensor(imgs)


    def __getitem__(self, index):

        should_get_same_class = self.prob.rvs()
        md1, md2, points, root = self.get_point(index)
          
        p1 = np.array(glob.glob("{}/{}/{}* {}* {}*/*.png".format(self.data_sketch_root+root, 
                                                                 md1, 
                                                                 '%.3f'%float(points[0]), 
                                                                 '%.3f'%float(points[1]), 
                                                                 '%.3f'%float(points[2]))))
        
        p2 = np.array(glob.glob("{}/{}/{}* {}* {}*/*.png".format(self.data_sketch_root+root, 
                                                                 md2, 
                                                                 '%.3f'%float(points[3]), 
                                                                 '%.3f'%float(points[4]), 
                                                                 '%.3f'%float(points[5]))))
        
        md1, md2, points, root = self.get_point(random.randint(0, self.__len__()-1))
        p3 = np.array(glob.glob("{}/{}/{}* {}* {}*/*.png".format(self.data_sketch_root+root, 
                                                                 md2, 
                                                                 '%.3f'%float(points[3]), 
                                                                 '%.3f'%float(points[4]), 
                                                                 '%.3f'%float(points[5]))))

        #vista 1, 3 y 6, sino descartar!.
        
        p1 = np.array(list(filter(lambda l: (l.split("/")[-1]).split(".")[0] in ["view_1", "view_3", "view_6"], p1)))
        p2 = np.array(list(filter(lambda l: (l.split("/")[-1]).split(".")[0] in ["view_1", "view_3", "view_6"], p2)))
        p3 = np.array(list(filter(lambda l: (l.split("/")[-1]).split(".")[0] in ["view_1", "view_3", "view_6"], p3)))
        
        if p1.shape[0] == 0 or p2.shape[0] ==0 or p3.shape[0]==0:
            index = random.randint(0,self.n-1)
            imgs_p0, imgs_p1 , imgs_p3  = self.__getitem__(index)
            return imgs_p0, imgs_p1, imgs_p3,
        
        
        p1 = np.random.choice(p1, 1)[0]
        p2 = np.random.choice(p2, 1)[0]
        p3 = np.random.choice(p3, 1)[0]
        
       
        return self.zooms(p1), self.zooms(p2), self.zooms(p3)
    
    def __len__(self):
        return self.n