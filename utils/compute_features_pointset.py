import sys
import utils.util as util
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
from os import path
import torchvision.transforms as transforms
from utils.Pca import pca_compute
from sklearn.decomposition import PCA

import utils.ShapeContextMod as sc
from numpy.linalg import norm
from utils.Gabor import gabor, gabor_compute

def compute_features_alexnet(m_A, path_A, netname="../net/checkpoints_mix/current_batch_checkpoints_mix.pkl", nettype="triplet", replace=False, device='cuda'):
    
    if path.exists(path_A.replace(".png", "_sketchzooms_alexnet_{}.npy".format(nettype))) and not replace:
        return np.load(path_A.replace(".png", "_sketchzooms_alexnet_{}.npy".format(nettype)))
    
    transform_torch= transforms.Compose([transforms.RandomRotation((0,0), fill=(255, 255, 255, 1)),
                                        transforms.RandomHorizontalFlip(p=0),
                                        transforms.Resize((224, 224)), 
                                        transforms.ToTensor()])
    
    net = util.init_net(netname, device)
    image_A = Image.open(path_A).convert('RGBA').resize((512, 512))
    m_A = np.array(m_A)

    fA = []
    if(m_A.shape[0] > 128):
        for i, j in zip(range(0, np.array(m_A).shape[0], 128), range(128, np.array(m_A).shape[0], 128)):
            #print(i, j)
            features_A = util.compute_feature(image_A, m_A[i:j], transform_torch= transform_torch )
            fA.append(util.compute_net(features_A, net, device))
            
        features_A = util.compute_feature(image_A, m_A[j:], transform_torch= transform_torch )
    else:
        features_A = util.compute_feature(image_A, m_A[:], transform_torch= transform_torch )

    f_last = util.compute_net(features_A, net, device)
    fba = np.array(fA[:]).reshape(-1, 128)
    fba = np.concatenate([fba, f_last])
    np.save(path_A.replace(".png", "_sketchzooms_alexnet_{}.npy".format(nettype)), fba)
    del net, image_A, f_last, fA
    return fba
'''
def compute_features_vgg(m_A, path_A, netname="", nettype="vgg", replace=False, device='cuda'):
    
    if path.exists(path_A.replace(".png", "_sketchzooms_vgg_{}.npy".format(nettype))) and not replace:
        return np.load(path_A.replace(".png", "_sketchzooms_vgg_{}.npy".format(nettype)))
    
    transform_torch= transforms.Compose([transforms.RandomRotation((0,0), fill=(255, 255, 255, 1)),
                                        transforms.RandomHorizontalFlip(p=0),
                                        transforms.Resize((224, 224)), 
                                        transforms.ToTensor()])
    
    net = util.init_net_vgg(netname, device)
    image_A = Image.open(path_A).convert('RGBA').resize((512, 512))
    m_A = np.array(m_A)

    fA = []
    if(m_A.shape[0] > 128):
        for i, j in zip(range(0, np.array(m_A).shape[0], 128), range(128, np.array(m_A).shape[0], 128)):
            #print(i, j)
            features_A = util.compute_feature(image_A, m_A[i:j], transform_torch= transform_torch )
            fA.append(util.compute_net(features_A, net, device))
            
        features_A = util.compute_feature(image_A, m_A[j:], transform_torch= transform_torch )
    else:
        features_A = util.compute_feature(image_A, m_A[:], transform_torch= transform_torch )

    f_last = util.compute_net(features_A, net, device)
    fba = np.array(fA[:]).reshape(-1, 128)
    fba = np.concatenate([fba, f_last])
    np.save(path_A.replace(".png", "_sketchzooms_vgg_{}.npy".format(nettype)), fba)
    del net, image_A, f_last, fA
    return fba
'''
def compute_features_vgg(m_A, path_A, netname="", nettype="vgg", replace=False, device='cuda'):
    
    if path.exists(path_A.replace(".png", "_sketchzooms_vgg_{}.npy".format(nettype))) and not replace:
        return np.load(path_A.replace(".png", "_sketchzooms_vgg_{}.npy".format(nettype)))
    
    transform_torch= transforms.Compose([transforms.RandomRotation((0,0), fill=(255, 255, 255, 1)),
                                        transforms.RandomHorizontalFlip(p=0),
                                        transforms.Resize((224, 224)), 
                                        transforms.ToTensor()])
    
    net = util.init_net_vgg(netname, device)
    image_A = Image.open(path_A).convert('RGBA').resize((512, 512))
    m_A = np.array(m_A)

    fA = []
    if(m_A.shape[0] > 64):
        for i, j in zip(range(0, np.array(m_A).shape[0], 64), range(64, np.array(m_A).shape[0], 64)):
            #print(i, j)
            features_A = util.compute_feature(image_A, m_A[i:j], transform_torch= transform_torch )
            fA.append(util.compute_net(features_A, net, device))
            
        features_A = util.compute_feature(image_A, m_A[j:], transform_torch= transform_torch )
    else:
        features_A = util.compute_feature(image_A, m_A[:], transform_torch= transform_torch )

    f_last = util.compute_net(features_A, net, device)
    fba = np.array(fA[:]).reshape(-1, 128)
    fba = np.concatenate([fba, f_last])
    np.save(path_A.replace(".png", "_sketchzooms_vgg_{}.npy".format(nettype)), fba)
    del net, image_A, f_last, fA
    return fba


def compute_features_resnet(m_A, path_A, netname="../net/checkpoints_mix/current_batch_checkpoints_mix.pkl", nettype="triplet",  replace=False, device='cuda'):
    
    if path.exists(path_A.replace(".png", "_sketchzoom_resnet_{}.npy".format(nettype))) and not replace:
        return np.load(path_A.replace(".png", "_sketchzoom_resnet_{}.npy".format(nettype)))
    
    transform_torch= transforms.Compose([transforms.RandomRotation((0,0), fill=(255, 255, 255, 1)),
                                        transforms.RandomHorizontalFlip(p=0),
                                        transforms.Resize((224, 224)), 
                                        transforms.ToTensor()])
    
    net = util.init_net_resnet(netname, device=device)
    image_A = Image.open(path_A).convert('RGBA').resize((512, 512))
    m_A = np.array(m_A)

    fA = []
    if(m_A.shape[0] > 128):
        for i, j in zip(range(0, np.array(m_A).shape[0], 128), range(128, np.array(m_A).shape[0], 128)):
            #print(i, j)
            features_A = util.compute_feature(image_A, m_A[i:j], transform_torch= transform_torch )
            fA.append(util.compute_net(features_A, net, device=device))
        features_A = util.compute_feature(image_A, m_A[j:], transform_torch= transform_torch)
    else:
        features_A = util.compute_feature(image_A, m_A[:], transform_torch= transform_torch )

    f_last = util.compute_net(features_A, net, device=device)
    fba = np.array(fA[:]).reshape(-1, 128)
    fba = np.concatenate([fba, f_last])
    np.save(path_A.replace(".png", "_sketchzoom_resnet_{}.npy".format(nettype)), fba)
    del net, image_A, f_last, fA, m_A
    return fba




def compute_features_shapecontex(m_A, path_A, name='', replace=False):
    
    if path.exists(path_A.replace(".png", "_SC_{}.npy".format(name))) and not replace:
        rtemp = np.load(path_A.replace(".png", "_SC{}.npy".format(name)))
        rtemp = rtemp/norm(rtemp, axis=1, ord=1).reshape(rtemp.shape[0],1)
        return rtemp
    
    st = sc.ShapeContext()
    rtemp = st.compute(path_A, m_A)
    rtemp = rtemp.reshape(m_A.shape[0], -1)
    np.save(path_A.replace(".png", "_SC{}.npy".format(name)), rtemp)
    rtemp = rtemp/norm(rtemp, axis=1, ord=1).reshape(rtemp.shape[0],1)
    return rtemp



def compute_features_gabor(m_A, path_A, replace=False):
    
    if path.exists(path_A.replace(".png", "_GABOR.npy")) and not replace :
        return np.load(path_A.replace(".png", "_GABOR.npy"))
    
    img_gabor_A = gabor(path_A)
    rtemp = gabor_compute(m_A, img_gabor_A)
    rtemp = rtemp.reshape(m_A.shape[0], -1)
    np.save(path_A.replace(".png", "_GABOR.npy"), rtemp)
    return rtemp

def compute_features_pca(m_A, path_A, pca = PCA(n_components=8), replace=False):
    
    if path.exists(path_A.replace(".png", "_PCA.npy")) and not replace :
        return np.load(path_A.replace(".png", "_PCA.npy")), pca
    
    rtemp, pca = pca_compute(m_A, path_A, pca)
    rtemp = rtemp.reshape(m_A.shape[0], -1)
    #np.save(path_A.replace(".png", "_PCA.npy"), rtemp)
    return rtemp, pca
    
