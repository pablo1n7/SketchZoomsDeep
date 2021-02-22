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
from utils.utilities import *

dic_indices = {0: 4, 1:5, 2:8}


class SketchNetworkDatasetTrain(Dataset): 
    
    def __init__(self, data_sketch_root, net, 
                 plotter, n=1000, transform=None, 
                 visdom=True, model_train="", 
                 image_size=224, triplet=True):
        
        self.corr_files = []
        self.visdom = visdom
        self.data_sketch_root = data_sketch_root
        self.transform = transform
        self.n = n
        self.n_all = []
        self.df = []
        self.views = []
        self.image_size= image_size
        self.triplet = (triplet * 4) + 1 
        for i in range(3):
            for j in range(3):
                self.views.append([dic_indices.get(i), dic_indices.get(j)])
                self.corr_files.append("{}/dataset_{}_train_{}_{}.csv".format(data_sketch_root, model_train, i, j))
                self.df.append(pd.read_csv(self.corr_files[-1], index_col=0))
                self.n_all.append(self.df[-1].shape[0])
        
        
        self.prob = stats.rv_discrete(name='custm', values=([0, 1], [0.95, 0.05]))
        self.plotter = plotter
        self.n_part_len = np.array(self.n_all) // self.n
        self.part = np.zeros(9)
        self.prob_rotations = stats.rv_discrete(name='custm', values=([0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25]))
        self.rotations = [lambda x : x, np.rot90, 
                          lambda x: np.rot90(np.rot90(x)), 
                          lambda x: np.rot90(np.rot90(np.rot90(x)))]
        self.net = net
        self.prob_show_image = stats.rv_discrete(name='custm', values=([0, 1], [0.95, 0.05]))
        self.prob_pixear_image = stats.rv_discrete(name='custm', values=([0, 1, 2], [0.60, 0.20, 0.20]))
        self.prob_view = stats.rv_discrete(name='custm', values=(np.arange(9), np.ones(9) * 1/9 ))
        self.prob_view_negative = stats.rv_discrete(name='custm', values=(np.arange(9), np.ones(9) * 1/9 ))
        self.pixear = [lambda x : x, 
                       lambda x: cv2.resize(x, (64, 64), interpolation=cv2.INTER_CUBIC), 
                       lambda x: cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)]
        self.init_csv()
        
    
    def init_csv(self):
        for i in range(9):
            self._init_csv(i)

    def _init_csv(self, indx):
        n_part_len = self.n_part_len[indx]
        part = self.part[indx]
        n_all = self.n_all[indx]
        corr_files = self.corr_files[indx]
        self.df[indx] = pd.read_csv(corr_files, header=0, 
                                    index_col=0, sep=",", 
                                    skiprows=int(part), nrows=int(self.n))
        
        if self.df[indx].shape[0] != self.n:
            self.part[indx] = 0
            self.n_part_len[indx] = n_all//self.n
            
            _df = pd.read_csv(corr_files, header=0, index_col=0, sep=",")
            _df = _df.sample(frac=1).reset_index(drop=True)
            _df.to_csv(corr_files)
            del _df
            
            self.df[indx] = pd.read_csv(corr_files, header=0, 
                                    index_col=0, sep=",", 
                                    skiprows=int(self.part[indx]), nrows=int(self.n))
            
        
        
        self.df[indx].reset_index(drop=True, inplace=True)
        self.part[indx] = self.part[indx] + self.n
        self.n_part_len[indx] = self.n_part_len[indx] -1
        

    def get_point(self, index, prob):
        indx = prob.rvs()
        if(self.df[indx].shape[0]<= index):
            index = random.randint(0, self.df[indx].shape[0]-1)
            
        points_df = self.df[indx].loc[[index]]
        row = np.array(points_df).flatten()
        md1 = row[0]
        md2 = row[1]
        points = np.array(row[2:],dtype=str)
        return md1, md2, points, self.views[indx]
    
    
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
        #print(len(img_points))
        image_original = img_points[0]
        image_original = Image.open(image_original)
        image_original = image_original.convert('RGB')
        image_original = np.array(image_original)
        image_original = Image.fromarray(normalize(image_original).astype('uint8'),'RGB')
        image_original = np.array(image_original)
        imgs = []
        #imgs.append(image_original)
        for z in [1.5, 1, 0.5]:
            z = z + np.random.uniform(-0.3, 0.3)
            img = get_windows(image_original, 256, 256, int(100 * z))
            img = self.rotations[self.prob_rotations.rvs()](img)
            img = self.pixear[self.prob_pixear_image.rvs()](img)
            imgs.append(img)
        return self.img_to_tensor(imgs)

    
    def __getitem__(self, index):

        should_get_same_class = self.prob.rvs()
        md1, md2, points, views = self.get_point(index, self.prob_view)
          
        p1 = np.array(glob.glob("{}/{}/{}* {}* {}*/*.png".format(self.data_sketch_root, 
                                                                 md1, 
                                                                 '%.3f'%float(points[0]), 
                                                                 '%.3f'%float(points[1]), 
                                                                 '%.3f'%float(points[2]))))
        
        if not should_get_same_class:
            p2 = np.array(glob.glob("{}/{}/{}* {}* {}*/*.png".format(self.data_sketch_root, 
                                                                     md2, 
                                                                     '%.3f'%float(points[3]), 
                                                                     '%.3f'%float(points[4]), 
                                                                     '%.3f'%float(points[5]))))
        else:
            p2 = np.array(glob.glob("{}/{}/{}* {}* {}*/*.png".format(self.data_sketch_root, 
                                                                     md1, 
                                                                     '%.3f'%float(points[0]), 
                                                                     '%.3f'%float(points[1]), 
                                                                     '%.3f'%float(points[2]))))
        
        p1 = np.array(list(filter(lambda l: (l.split("/")[-1]).split(".")[0] in ["view_{}".format(views[0])], p1)))
    
        p2 = np.array(list(filter(lambda l: (l.split("/")[-1]).split(".")[0] in ["view_{}".format(views[1])], p2)))

        if p1.shape[0] == 0 or p2.shape[0] ==0:            
            index = random.randint(0,self.n-1)
            imgs_p0, imgs_p1 , imgs_p4 = self.__getitem__(index)
            return imgs_p0, imgs_p1, imgs_p4
        
        p1_img = self.zooms(p1)
        p2_img = self.zooms(p2)
        img1, img2 = Variable(p1_img).cuda(0), Variable(p2_img).cuda(0) 
        p3_img = None
        for i in range(self.triplet):
            p3 = np.array([])
            while (p3.shape[0] == 0):
                md1, md2, points, views = self.get_point(random.randint(0, self.__len__()-1),
                                                         self.prob_view_negative)
                
                p3 = np.array(glob.glob("{}/{}/{}* {}* {}*/*.png".format(self.data_sketch_root, 
                                                                         md2, 
                                                                         '%.3f'%float(points[3]), 
                                                                         '%.3f'%float(points[4]), 
                                                                         '%.3f'%float(points[5]))))
                p3 = np.array(list(filter(lambda l: (l.split("/")[-1]).split(".")[0] in ["view_{}".format(views[0])], p3)))
            
            p3_img = self.zooms(p3)
            img3 = Variable(p3_img).cuda(0)
            
            if self.triplet > 1:
                anchor, positive, negative = self.net(img1.view(-1, 3, 3, self.image_size, self.image_size),
                                                      img2.view(-1, 3, 3, self.image_size, self.image_size),
                                                      img3.view(-1, 3, 3, self.image_size, self.image_size), 1)

                distance_positive = (anchor - positive).pow(2).sum(1).data.cpu()[0]
                distance_negative = (anchor - negative).pow(2).sum(1).data.cpu()[0]
                if ((distance_positive < distance_negative) and (distance_negative < (distance_positive + 1))):
                    break 

        if(self.prob_show_image.rvs() == 1 and self.visdom):
            images = torch.cat([p1_img, p2_img, p3_img]).view(9 ,3, self.image_size, self.image_size)
            self.plotter.images(images)
            
            
        return p1_img, p2_img, p3_img
    
    def __len__(self):
        return self.n
