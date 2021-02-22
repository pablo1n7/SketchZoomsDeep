from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
sys.path.append('../.')
from net.SketchNetwork import SketchNetwork
from net.SketchNetworkResnet import SketchNetworkResnet
from net.SketchNetworkVGG import SketchNetworkVGG
from utils.utilities import *
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt
from scipy import stats


    
def subdivide(gray, t, x0, y0, size=128, image_size=512, nivel=0):
    img = gray
    gray = np.array(gray)
    coord_select=[]
    t = 2**t
    x0 = int(x0)
    y0 = int(y0)
    r = image_size // t
    n = r*r
    frag_select = []
    x = 0
    y = 0
    for i in range(r):
        for j in range(r):
            #print(x,x+size,y,y+size)
            #if ((gray[x:x+size, y:y+size].mean() != 255)):
            d = np.array(img.crop((x+x0, y+y0, x+size+x0, y+size+y0)))
            d[d[:, :, 3] == 0] = [255, 255, 255, 1]
            d = d[:, :, [0, 1, 2]]
            img_crop = d
            
            if np.array(d).mean() < 253:
                
                #cv2.imwrite("test/test{}{}.png".format(i, j), d)

                coord_select.append([[x + x0, -1, x+size+x0],[y+y0, -1 ,y+size+y0]])
                frag_select = frag_select + compute_zooms(img_crop, nivel)    
            y = y + t
        x = x + t
        y = 0
    return coord_select, img_to_tensor(frag_select, transform=transforms.Compose([transforms.Scale((224,224)),
                                                                                  transforms.ToTensor()])), frag_select

def compute_net(features_A, net, device='cuda'):
    features = Variable(features_A).view(-1, 3, 3, 224, 224).to(device)
    output_1 = net.forward_one(features, features.size()[0])
    output_1 = output_1.data.cpu().numpy()
    return output_1

def compute_pair_net(features_A, features_B, net):
    features = Variable(torch.cat((features_A, features_B), 0))
    features = features.view((-1, 3, 3, 224, 224)).cuda()
    output_1 = net.forward_one(features, features.size()[0])
    output_1 = output_1.data.cpu().numpy()
    return output_1[:features_A.size()[0]//9], output_1[features_A.size()[0]//9:]


def compute_pair_net_batch(features_A, features_B, net):
    print(features_A.size(), features_B.size())
    features = Variable(torch.cat((features_A, features_B), 0))
    features = features.view((-1, 3, 3, 224, 224)).cuda()
    print(features.size())
    return
    output_1 = net.forward_one(features, features.size()[0])
    output_1 = output_1.data.cpu().numpy()
    return output_1[:features_A.size()[0]//9], output_1[features_A.size()[0]//9:]


def init_net(net_file, device='cuda'):
    net = SketchNetwork()
    checkpoint = torch.load(net_file, map_location=torch.device(device))
    if list(checkpoint['model_state_dict'].keys())[0].split('.')[0] == 'module':
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.module
    else:
        net.load_state_dict(checkpoint['model_state_dict'])
    
    net = net.to(device)
    net.eval()
    
    return net

def init_net_resnet(net_file, device='cuda'):
    net = SketchNetworkResnet()
    checkpoint = torch.load(net_file, map_location=torch.device(device))
    if list(checkpoint['model_state_dict'].keys())[0].split('.')[0] == 'module':
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.module
    else:
        net.load_state_dict(checkpoint['model_state_dict'])
    
    net = net.to(device)
    net.eval()
    return net

def init_net_vgg(net_file, device='cuda'):
    net = SketchNetworkVGG()
    checkpoint = torch.load(net_file, map_location=torch.device(device))
    if list(checkpoint['model_state_dict'].keys())[0].split('.')[0] == 'module':
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.module
    else:
        net.load_state_dict(checkpoint['model_state_dict'])
    
    net = net.to(device)
    net.eval()
    return net

def match_one(a, fB):
    distancias = []
    for b in fB:
        distancias.append(distance.euclidean(a, b))
    return np.argmin(distancias)   

def match_3(a, fB):
    distancias = []
    for b in fB:
        distancias.append(distance.euclidean(a, b))
    index_sorted = sorted(range(len(distancias)), key=lambda k: distancias[k])
    return index_sorted[0:1]
    
def match_one_dist(a, fB):
    distancias = []
    for b in fB:
        distancias.append(distance.euclidean(a, b))
    return np.argmin(distancias), np.min(distancias)   


def match_cv2(fA, fB, crossCheck=True):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    # Match descriptors.
    matches = bf.match(np.asarray(fA,np.float32),np.array(fB,np.float32))
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    matches_fA = list(map(lambda x: x.queryIdx, matches))
    matches_fB = list(map(lambda x: x.trainIdx, matches))
    return matches_fA, matches_fB

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

def match_linear_sum_assignment(fA, fB, crossCheck=True):
    cost = np.zeros((fA.shape[0],fB.shape[0]),dtype=np.float)
    for i, out_1 in enumerate(fA):
        for j, out_2 in enumerate(fB):
            dst = distance.euclidean(out_1, out_2)
            cost[i][j]= dst
    
    matches_fA, matches_fB = linear_sum_assignment(cost)
    return matches_fA, matches_fB
