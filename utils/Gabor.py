import cv2
import numpy as np
from PIL import Image


def build_filters():
    filters = []
    ksize = 8
    for theta in np.arange(0, np.pi, np.pi / 4):
        kern = cv2.getGaborKernel((ksize, ksize), 0.13, theta, 0.3, 0.02, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters
 
def process(img, filters):
    accum = np.zeros_like(img)
    images = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_32FC3, kern)
        images.append(fimg[:,:, 0])
    return images

def gabor(img_fn):
    img = cv2.imread(img_fn)
    img = cv2.resize(img,(512,512))
    filters = build_filters()
    img = np.array(img, dtype=np.float32)
    img = img #/ 255
    res1 = process(img, filters)
    return res1


def other_method_fit(_a_point_2d_v1, img_o_array, size=64):
    X = []
    img_o = Image.fromarray(img_o_array)
    img = Image.new('L', (1024, 1024), 255)
    img.paste(img_o, (256, 256))
    for xy in _a_point_2d_v1:
        hwq = (xy[0] - size + 256, 
               xy[1] - size + 256, 
               xy[0] + size + 256, 
               xy[1] + size + 256)
        img_x = img.crop(hwq)
        X.append(np.array(img_x))
        #imshow_both(img_x, img, [[size, size]], [np.array(xy) + 256, [hwq[0], hwq[1]], [hwq[2], hwq[3]]])
        
    X = np.array(X).reshape(len(_a_point_2d_v1), -1)
    return X

def gabor_compute(_a_point_2d_v1, g_img):
    X = []
    step = 16
    for xy in _a_point_2d_v1:
        x = []
        for g in g_img:
            #x.append(np.array(g[xy[1] - step :xy[1] + step , xy[0] - step : xy[0] + step]))
            x.append(other_method_fit([xy], g, step))
        X.append(np.array(x).flatten())
    X = np.array(X).reshape(_a_point_2d_v1.shape[0], -1)
    return X
