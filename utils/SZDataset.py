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


class SketchZoomDataset(Dataset): 
    
    def __init__(self,
                data_sketch_root, 
                net, 
                plotter, 
                n=1000, 
                transform=None, 
                visdom=None,
                categories=[],
                stage="", 
                image_size=224, 
                triplet=True,
                device="cuda"):
        
        self.corr_files = []
        self.visdom = visdom
        self.data_sketch_root = data_sketch_root
        self.transform = transform
        self.n = n
        #self.n_all = []
        #self.views = []
        #self.df = []
        self.image_size= image_size
        self.triplet = (triplet * 4)
        self.device = device
        self.df = pd.read_csv("{}csv/{}.csv".format(data_sketch_root, stage), index_col=0)
        '''
        for i in range(3):
            for j in range(3):
                self.views.append([dic_indices.get(i), dic_indices.get(j)])
                
                self.corr_files.append("{}csv/{}_{}_{}.csv".format(data_sketch_root, stage, i, j))
                self.df.append(pd.read_csv(self.corr_files[-1], index_col=0))
                self.n_all.append(self.df[-1].shape[0])
        '''
        self.plotter = plotter
        #self.n_part_len = np.array(self.n_all) // self.n
        #self.part = np.zeros(9)
        #self.categories = categories
        #self.categories_prob = stats.rv_discrete(name='custm', values=(np.arange(len(categories)), 
        #                                                               np.ones(len(categories)) * 1/len(categories) ))
        self.net = net
        self.prob_show_image = stats.rv_discrete(name='custm', values=([0, 1], [0.95, 0.05]))
        
        #self.prob_view = stats.rv_discrete(name='custm', values=(np.arange(9), np.ones(9) * 1/9 ))
        #self.prob_view_negative = stats.rv_discrete(name='custm', values=(np.arange(9), np.ones(9) * 1/9 ))
        
        #self.init_csv()
        
    ''' 
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
    ''' 
            
    def zooms(self, image_original):
        z = np.array([1.8, 1.3, 0.8]) - np.random.uniform(-0.3, 0.3)
        image_original = Image.open(image_original).convert('RGBA')
        image_original = np.array(image_original)
        #image_original = Image.fromarray(normalize(image_original).astype('uint8'),'RGBA')
        image_original = Image.fromarray(image_original.astype('uint8'),'RGBA')
        return compute_feature(image_original, transform_torch=self.transform, z_zoom=z)

    def get_point(self, index, prob, category):
        '''
        indx = prob.rvs()
        if(self.df[indx].shape[0]<= index):
            index = random.randint(0, self.df[indx].shape[0]-1)
        '''    
        #points_df = self.df[indx][self.df[indx]['model'] == category].sample(1).to_dict('records')[0]
        points_df = self.df.iloc[index]
        md1 = points_df["model_0"]
        md2 = points_df["model_1"]
        points = [points_df["index_0"], points_df["index_1"]]
        return md1, md2, points, points_df["view"].split(' '), points_df["model"], 
        #return md1, md2, points, self.views[indx], points_df["model"], 
        
    def get_point_category(self, index, category):
        
        sample = self.df[self.df['model'] == category].sample(1)
        while (sample.index[0] == index):
            sample = self.df[self.df['model'] == category].sample(1)
        
        points_df= sample.to_dict('records')[0]
        #points_df = self.df.iloc[index]
        md1 = points_df["model_0"]
        md2 = points_df["model_1"]
        points = [points_df["index_0"], points_df["index_1"]]
        return md1, md2, points, points_df["view"].split(' '), points_df["model"], 
    
    def __getitem__(self, index):
        #catx = self.categories_prob.rvs()
        #md1, md2, points, views, model = self.get_point(index, self.prob_view, self.categories[catx])
        md1, md2, points, views, model = self.get_point(index, None, '')
        
        p1 = np.array(glob.glob("{}/pts_{}/{}/{}/view_{}.png".format(self.data_sketch_root,
                                                              model,
                                                              md1,points[0], 
                                                              views[0])))[0]
        
        p2 = np.array(glob.glob("{}/pts_{}/{}/{}/view_{}.png".format(self.data_sketch_root, 
                                                                     model,
                                                                     md2,
                                                                     points[1], 
                                                                     views[1])))[0]
        
        p1_img = self.zooms(p1)
        p2_img = self.zooms(p2)
        
        img1, img2 = p1_img.to(self.device), p2_img.to(self.device)
        
        p3_img = None
        self.net.eval() 
        for i in range(self.triplet+1):
        
            md1, md2, points, views, _ = self.get_point_category(index, model)       
            p3 = np.array(glob.glob("{}/pts_{}/{}/{}/view_{}.png".format(self.data_sketch_root, 
                                                                         model, 
                                                                         md2,points[1], 
                                                                         views[1])))[0]
            p3_img = self.zooms(p3)
            #img3 = Variable(p3_img).to(self.device)
            img3 = p3_img.to(self.device)
            if self.triplet > 1:
                with torch.no_grad():
                    anchor, positive, negative = self.net(img1.view(-1, 3, 3, self.image_size, self.image_size),
                                                          img2.view(-1, 3, 3, self.image_size, self.image_size),
                                                          img3.view(-1, 3, 3, self.image_size, self.image_size), 1)

                    distance_positive = (anchor - positive).pow(2).sum(1).data.cpu()[0]
                    distance_negative = (anchor - negative).pow(2).sum(1).data.cpu()[0]
                    
                if ((distance_positive < distance_negative) and (distance_negative < (distance_positive + 1))):
                    break 

        if(self.prob_show_image.rvs() == 1):
            if (self.visdom is not None):
                images = torch.cat([p1_img, p2_img, p3_img]).view(9 ,3, self.image_size, self.image_size)
                self.plotter.images(images)

        return p1_img, p2_img, p3_img
    
    def __len__(self):
        return self.n
