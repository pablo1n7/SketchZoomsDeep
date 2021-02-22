from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
import os

class SketchNetwork(nn.Module):
    def __init__(self, size=224, size_feature=128):
        super(SketchNetwork, self).__init__()
        self.size = size
        self.size_feature = size_feature
        original_model = models.alexnet(pretrained=False) #set in True if you'll reatrain 
        self.features = nn.Sequential(
            *list(original_model.features.children())
        )
        self.classifier = nn.Sequential(
            *list(original_model.classifier.children())[:-4],
            
        )
        self.dimreduction = nn.Sequential(
            nn.Linear(4096, self.size_feature)
        )
        self.class_binary = nn.Sequential(
            nn.Linear(self.size_feature*2, self.size_feature),
            nn.Softmax(dim=0),
            nn.Linear(self.size_feature, self.size_feature),
            nn.ReLU(),
            nn.Linear(self.size_feature, 2),
            nn.Softmax(dim=1)
        )

    def forward_once(self, x):
        output = self.features(x)
        output = output.view(x.size(0), 256 * 6 * 6)
        output = self.classifier(output)
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
    
    def forward_two_binary(self, input_sketch, input_normal, batch_size):    
        input_cat = torch.cat((input_sketch, input_normal))
        output = self.forward_full(input_cat).view(-1, input_sketch.size()[0], self.size_feature)
        output_ = output.permute((1,0,2))
        result = self.class_binary(output_.reshape((input_sketch.size()[0], -1)))        
        return output[0], output[1], result
    
    def forward_one(self, input_sketch, batch_size):
        output = self.forward_full(input_sketch).view(-1, input_sketch.size()[0], self.size_feature)
        return output[0]
