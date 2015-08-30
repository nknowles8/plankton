__author__ = 'Sachith Thomas and Nick Knowles'

import numpy as np
import glob
import os
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
from pylab import cm
from skimage import morphology, measure


DIR = "data/train/"


def threshold_image(im):
    """
    threshold image using its mean pixel val
    :param im: image in the form of 2d numpy array
    :return: image in the form of 2d numpy array consisting of 0s and 1s representing image mask
    """
    imthr = np.where(im > np.mean(im), 0.0, 1.0)
    return imthr


def dilate_image(im, n=4):
    """
    dilate an image by setting each pixel to the max val within an nxn filter
    :param im: image in the form of 2d numpy array
    :return: dilated image in the form of 2d numpy array
    """
    imdilated = morphology.dilation(im, np.ones((n, n)))
    return imdilated


def segment_image(im):
    """
    label and separate connecting regions in image using thresholding and dilation
    :param im: 2d numpy array
    :return: labelled 2d numpy array
    """
    imthr = threshold_image(im)
    imdilated = dilate_image(imthr)
    labels = measure.label(imdilated)
    labels = imthr*labels
    labels = labels.astype(int)
    return labels


def test():
    dir_names = glob.glob(os.path.join("data", "train", "*"))
    ex_file = glob.glob(os.path.join(dir_names[5], "*.jpg"))[5]
    print ex_file
    im = imread(ex_file, as_grey=True)
    labels = segment_image(im)
    regions = measure.regionprops(labels)
    print type(regions)

if __name__ == "__main__":
    test()