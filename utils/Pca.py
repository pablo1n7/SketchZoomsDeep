import cv2
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import AxesGrid
import glob
from PIL import Image
import cv2
import numpy as np

def other_method_fit(_a_point_2d_v1, img_o, pca, size=64):
    X = []
    img = Image.new('L', (1024, 1024), 255)
    img.paste(img_o, (256, 256))
    for xy in _a_point_2d_v1:
        hwq = (xy[0] - size + 256, 
               xy[1] - size + 256, 
               xy[0] + size + 256, 
               xy[1] + size + 256)
        #print(np.array(hwq))
        img_x = img.crop(hwq)
        X.append(np.array(img_x))
        #imshow_both(img_x, img, [[size, size]], [np.array(xy) + 256, [hwq[0], hwq[1]], [hwq[2], hwq[3]]])
        
    X = np.array(X).reshape(len(_a_point_2d_v1), -1)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def pca_compute(m_A, path_A, pca):
    img = Image.open(path_A).convert('L').resize((512, 512))
    rtemp, pca = other_method_fit(m_A, img, pca)
    rtemp = rtemp.reshape(m_A.shape[0], -1)
    return rtemp, pca
        