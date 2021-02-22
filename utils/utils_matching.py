import matplotlib.pyplot as plt
from os import path
import cv2
import numpy as np
from utils.sampling import get_sampling_coords
import torchvision.transforms as transforms

def imshow_scatter(img1, points_1, msk, marker_size=30):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(*zip(*points_1), alpha=0.5, s=marker_size)

    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(msk)
    plt.xticks([])
    plt.yticks([])

    plt.show()
    
    
from scipy.spatial import distance

def match_m(img1, img2, x_y_img_1, x_y_img_2, origin_inx, costs, ft_img_1, ft_img_2, normalized=True):
    
    # red pair
    origin_single_point = ft_img_1[origin_inx]
    distance_single = list(map(lambda x: distance.euclidean(origin_single_point, x), ft_img_2)) 
    target_single_point = x_y_img_2[np.argsort(distance_single)][0]
    return target_single_point
    

def imshow_both_interp(img1, img2, x_y_img_1, x_y_img_2, origin_inx, costs, ft_img_1, ft_img_2, normalized=True):
    
    # red pair
    origin_single_point = [x_y_img_1[origin_inx]]
    target_single_point = x_y_img_2[np.argsort(costs[origin_inx])[:1]]

    # compute distances 
    dist_origin = np.inner(ft_img_1, ft_img_1)
    normalized_dist_origin = dist_origin[origin_inx]
    normalized_dist_target = costs[origin_inx]
    
    # Change scale, better non-linear
    if normalized:
        normalized_dist_origin = (dist_origin[origin_inx] - dist_origin[origin_inx].min()) / (dist_origin[origin_inx].max() - dist_origin[origin_inx].min())
        normalized_dist_target = (costs[origin_inx] - costs[origin_inx].min()) / (costs[origin_inx].max() - costs[origin_inx].min())
        normalized_dist_origin = normalized_dist_origin**(2)
        normalized_dist_target = normalized_dist_target**(1/2)
    
    # plot all dots
    # left
    plt.figure(figsize=(12,5))
    plt.title("Linear graph")
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(*zip(*x_y_img_1),alpha=0.2,c=normalized_dist_origin,cmap='plasma',linewidth=0)
    plt.scatter(*zip(*origin_single_point),color='red',alpha=0.8, marker='x')

    plt.xticks([])
    plt.yticks([])

    # right
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(*zip(*x_y_img_2),alpha=0.2,c=normalized_dist_target,cmap='plasma_r',linewidth=0)
    plt.scatter(*zip(*target_single_point),color='red',alpha=0.8,  marker='x')

    plt.xticks([])
    plt.yticks([])

    plt.show()

def imshow_both_match(img1, img2, x_y_img_1, x_y_img_2, origin_inx, costs, ft_img_1, ft_img_2, normalized=True):
    
    # red pair
    origin_single_point = [x_y_img_1[origin_inx]]
    target_single_point = x_y_img_2[np.argsort(costs[origin_inx])[:1]]

    # compute distances 
    dist_origin = np.inner(ft_img_1,ft_img_1)
    normalized_dist_origin = (dist_origin[origin_inx] - dist_origin[origin_inx].min()) / (dist_origin[origin_inx].max() - dist_origin[origin_inx].min())
    normalized_dist_target = (costs[origin_inx] - costs[origin_inx].min()) / (costs[origin_inx].max() - costs[origin_inx].min())

    # Change scale, better non-linear
    if normalized:
        normalized_dist_origin = normalized_dist_origin**(2)
        normalized_dist_target = normalized_dist_target**(1/2)
    
    # plot all dots
    # left
    plt.figure(figsize=(12,5))
    plt.title("Linear graph")
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    #plt.scatter(*zip(*x_y_img_1),alpha=0.2,c=normalized_dist_origin,cmap='plasma',linewidth=0)
    plt.scatter(*zip(*x_y_img_1[0:4]),alpha=1,linewidth=0)
    plt.scatter(*zip(*origin_single_point),color='red',alpha=0.8, marker='x')

    plt.xticks([])
    plt.yticks([])

    # right
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(*zip(*x_y_img_2[0:4]),alpha=1,linewidth=0)
    #plt.scatter(*zip(*x_y_img_2),alpha=0.2,c=normalized_dist_target,cmap='plasma_r',linewidth=0)
    plt.scatter(*zip(*target_single_point),color='red',alpha=0.8,  marker='x')

    plt.xticks([])
    plt.yticks([])

    plt.show()
    
def imshow_match(img1, img2, x_y_img_1, x_y_img_2, origin_inx_1, origin_inx_2 ):
    
    # red pair
    origin_single_point = [x_y_img_1[origin_inx_1]]
    target_single_point = [x_y_img_2[origin_inx_2]]

   
    plt.figure(figsize=(12,5))
    plt.title("Linear graph")
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    #plt.scatter(*zip(*x_y_img_1),alpha=0.2,c=normalized_dist_origin,cmap='plasma',linewidth=0)
    plt.scatter(*zip(*x_y_img_1[0:4]),alpha=1,linewidth=0)
    plt.scatter(*zip(*origin_single_point),color='red',alpha=0.8, marker='x')

    plt.xticks([])
    plt.yticks([])

    # right
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(*zip(*x_y_img_2[0:4]),alpha=1,linewidth=0)
    #plt.scatter(*zip(*x_y_img_2),alpha=0.2,c=normalized_dist_target,cmap='plasma_r',linewidth=0)
    plt.scatter(*zip(*target_single_point),color='red',alpha=0.8,  marker='x')

    plt.xticks([])
    plt.yticks([])

    plt.show()
    
def imshow_match_xyz(img1, img2, x_y_img_1, x_y_img_2, origin_inx_1, target_point ):
    
    # red pair
    origin_single_point = [x_y_img_1[origin_inx_1]]
   
    plt.figure(figsize=(12,5))
    plt.title("Linear graph")
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    #plt.scatter(*zip(*x_y_img_1),alpha=0.2,c=normalized_dist_origin,cmap='plasma',linewidth=0)
    plt.scatter(*zip(*x_y_img_1[0:4]),alpha=1,linewidth=0)
    plt.scatter(*zip(*origin_single_point),color='red',alpha=0.8, marker='x')

    plt.xticks([])
    plt.yticks([])

    # right
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(*zip(*x_y_img_2[0:4]),alpha=1,linewidth=0)
    #plt.scatter(*zip(*x_y_img_2),alpha=0.2,c=normalized_dist_target,cmap='plasma_r',linewidth=0)
    plt.scatter(target_point[0][0], target_point[0][1],  color='red', alpha=0.8,  marker='x', label='alexnet_triple')
    plt.scatter(target_point[1][0], target_point[1][1], color='blue', alpha=0.8,  marker='^', label='contrast')
    plt.scatter(target_point[2][0], target_point[2][1], color='green', alpha=0.8,  marker='o', label='entropy')
    plt.scatter(target_point[3][0], target_point[3][1], color='yellow',alpha=0.8,  marker='.', label='shapecontext')
    
    plt.legend()

    plt.xticks([])
    plt.yticks([])

    plt.show()
    
def imshow_msk(img,msk):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(msk, cmap="gray")
    plt.show()
    
def imshow(img,msk):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(msk)
    plt.show()
    
def imshow_sample(img1, points_1):
    plt.figure(figsize=(12,5))
    plt.imshow(img1)
    plt.scatter(*zip(*points_1),alpha=0.5)
    plt.show()
    
def imshow_both(img1,img2,points_1,points_2):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(*zip(*points_1),alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(*zip(*points_2),alpha=0.5)

    plt.show()


def get_sampling(path_img, r=5, replace=False):
    path_msk = path_img.replace(".png", "_mask.png")
    #imagen y mascara
    img = cv2.resize(cv2.imread(path_img),(512, 512))
    msk = cv2.resize(cv2.imread(path_msk),(512, 512))
    # binarizar la mascara
    msk_bool = msk == [0,0,0]
    msk_bool = msk_bool[:, :, 0]
    
    if not path.exists(path_img.replace(".png", "_sample.npy")) or replace:
        samples_img_tmp    = get_sampling_coords(msk_bool, r).round().astype(int)
        samples_img        = samples_img_tmp.copy()
        samples_img[:,0]   = samples_img_tmp[:,1]
        samples_img[:,1]   = samples_img_tmp[:,0]
        np.save(path_img.replace(".png", "_sample.npy"), samples_img)
        return img, msk_bool, samples_img
    else:
        return img, msk_bool, np.load(path_img.replace(".png", "_sample.npy"))
    

import ot

def get_otmatching(feature_sketchzoom_A, feature_sketchzoom_B):
    # Match
    # Cost matrix for matching
    # Optimal transport computation
    inner = np.inner(feature_sketchzoom_A, feature_sketchzoom_B)
    # Soft matching
    a = np.ones(feature_sketchzoom_A.shape[0])/feature_sketchzoom_A.shape[0]
    b = np.ones(feature_sketchzoom_B.shape[0])/feature_sketchzoom_B.shape[0]
    ot_m = ot.sinkhorn(a, b, inner, 1)
    
    return ot_m
