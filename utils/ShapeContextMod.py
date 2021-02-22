import numpy as np
import math
import cv2

def xrange(x, y=None):
    if y == None:
        return iter(range(x))
    return iter(range(x,y))

def radial_edges(r1, r2, n):
    #return a list of radial edges from an inner (r1) to an outer (r2) radius
    re = [ r1* ( (r2/r1)**(k /(n - 1.) ) ) for k in xrange(0, n)]
    return re

def euclid_distance(p1, p2):
    return math.sqrt( ( p2[0] - p1[0] ) ** 2 + ( p2[1] - p1[1] ) ** 2 )


def get_angle(p1, p2):
    #compute the angle between points.
    return math.atan2((p2[1] - p1[1]),(p2[0] - p1[0]))

from utils.utils_matching import imshow_msk, imshow_sample, imshow_scatter, get_sampling, imshow_both, imshow_both_interp, get_otmatching


class ShapeContext(object):
    """
    Given a point in the image, for all other points that are within a given
    radius, computes the relative angles.
    Radii and angles are stored in a  "shape matrix" with dimensions: radial_bins x angle_bins.
    Each element (i,j) of the matrix contains a counter/integer that corresponds to,
    for a given point, the number of points that fall at that i radius bin and at
    angle bin j.
    """

    def __init__(self,nbins_r=6,nbins_theta=12,r_inner=0.1250,r_outer=2.5,wlog=False):
        self.nbins_r        = nbins_r             # number of bins in a radial direction
        self.nbins_theta    = nbins_theta         # number of bins in an angular direction
        self.r_inner        = r_inner             # inner radius
        self.r_outer        = r_outer             # outer radius
        self.nbins          = nbins_theta*nbins_r # total number of bins
        self.wlog           = wlog                # using log10(r) or Normalize with the mean


    def distM(self, x_sampling, x_strokes):
        """
        Compute the distance matrix
        Params:
        -------
        x_sampling: a list with points tuple(x,y) to compute features in an image
        x_strokes:  a list with points tuple(x,y) with the actual data of the image
        Returns:
        --------
        result: a distance matrix with euclidean distance
        """

        result = np.zeros((len(x_sampling), len(x_strokes)))
        for i in xrange(len(x_sampling)):
            for j in xrange(len(x_strokes)):
                result[i, j] = euclid_distance(x_sampling[i], x_strokes[j])
        return result

    def angleM(self, x_sampling, x_strokes):
        """
        Compute the angle matrix
        Params:
        -------
        x_sampling: a list with points tuple(x,y) to compute features in an image
        x_strokes:  a list with points tuple(x,y) with the actual data of the image

        Returns:
        --------
        result: a matrix with angles among samples and strokes
        """

        result = np.zeros((len(x_sampling), len(x_strokes)))
        for i in xrange(len(x_sampling)):
            for j in xrange(len(x_strokes)):
                result[i, j] = get_angle(x_sampling[i], x_strokes[j])
        return result

    def compute(self, path_img, sampling_img_1):
        # Load and resize
        img_1 = cv2.resize(cv2.imread(path_img), (512, 512))
        msk_1 = cv2.resize(cv2.imread(path_img.replace(".png", "_mask.png")),(512, 512))
        #imshow_msk(img_1,msk_1)
        x, y = np.where(msk_1[:,:,0] != 255)
        xy = np.concatenate([x.reshape(1, -1), y.reshape(1, -1)], axis=0).T
        xy = xy[range(0, xy.shape[0], 3)]
        strokes_img_1 = []

        # Second, filter only those black points in the sketch
        for _xy in xy:
            if img_1[_xy[0]][_xy[1]][0] < 230:
                strokes_img_1.append(_xy)
        strokes_img_1 = np.array(strokes_img_1)[:,[1, 0]]
        
        #imshow_scatter(img_1, sampling_img_1, msk_1)
        #imshow_scatter(img_1,strokes_img_1,msk_1,1)
        
        # Now we are ready to compute SC
        # distance matrix, from sampling points to stroke points 
        r_array = self.distM(sampling_img_1,strokes_img_1)

        # Normalize the distance matrix by the mean distance or use log10
        if self.wlog:
            r_array_n = np.log10(r_array+1)
        else:
            mean_dist = r_array.mean()
            r_array_n = r_array / mean_dist
        # radial bins:
        r_bin_edges = radial_edges(self.r_inner, self.r_outer, self.nbins_r)
        # matrix with labels depending on the location of the points relative to each other
        r_array_bin = np.zeros((len(sampling_img_1),len(strokes_img_1)), dtype=int)
        for m in xrange(self.nbins_r):
            r_array_bin +=  (r_array_n < r_bin_edges[m])
        r_bool = r_array_bin > 0
        # angular matrix
        theta_array = self.angleM(sampling_img_1, strokes_img_1)

        # Ensure all angles are between 0 and 2Pi
        theta_array_2pi = theta_array + 2 * math.pi * (theta_array < 0)

        # from angle value to angle bin
        theta_array_bin = (1 + np.floor(theta_array_2pi /(2 * math.pi / self.nbins_theta))).astype('int')
        BH = np.zeros(len(sampling_img_1) * self.nbins)

        # For each sample point 
        for i in xrange(0,len(sampling_img_1)):
            # Create an empty SC feature
            sm = np.zeros((self.nbins_r, self.nbins_theta))
            # For all stroke points
            for j in xrange(len(strokes_img_1)):
                # If its in the scope (True = within radius of interest)
                if (r_bool[i, j]):
                    # if it is within radius, add 1 to the corresponding location in sm
                    sm[r_array_bin[i, j] - 1, theta_array_bin[i, j] - 1] += 1
            BH[i*self.nbins:i*self.nbins+self.nbins] = sm.reshape(self.nbins)
        
        sc_features = BH.reshape(sampling_img_1.shape[0], -1)
        return sc_features
