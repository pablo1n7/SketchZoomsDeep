from torch.autograd import Variable
from collections import OrderedDict
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class SketchNetworkInception(nn.Module):
    def __init__(self, size=299, size_feature=128):
        super(SketchNetworkInception, self).__init__()
        self.size = size
        self.size_feature = size_feature
        self.original_model = models.inception_v3(pretrained=True, transform_input=False)
        
        
        self.dimreduction = nn.Sequential(
            nn.Linear(1000, self.size_feature)
        )

    def forward_once(self, x):
        output = self.original_model(x)[0]
        return output   
    
    
    def forward_full(self, xs):
        xs = xs.view(-1, 3, self.size, self.size)
        xs_c = self.dimreduction(self.forward_once(xs)).view((-1, self.size_feature, 3))
        return xs_c.max(dim=2)[0]
           
    def forward(self, input_sketch, input_normal, input_negative, batch_size):
        input_cat = torch.cat((input_sketch, input_normal, input_negative))
        output = self.forward_full(input_cat).view(-1, input_sketch.size()[0], self.size_feature)
        return output[0], output[1], output[2]
    
    def forward_two(self, input_sketch, input_normal, batch_size):    
        input_cat = torch.cat((input_sketch, input_normal))
        output = self.forward_full(input_cat).view(-1, input_sketch.size()[0], self.size_feature)
        return output[0], output[1]
    
    def forward_one(self, input_sketch, batch_size):
        input_cat = torch.cat((input_sketch))
        output = self.forward_full(input_cat).view(-1, input_sketch.size()[0], self.size_feature)
        return output[0]