__author__ = 'Sachith Thomas and Nick Knowles'

import numpy as np
import glob
import os
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
from pylab import cm
from skimage import morphology, measure


def get_stats():
    dir_names = glob.glob(os.path.join("data", "train", "*"))
    total_files = 0
    total_width = 0
    total_height = 0
    max_width = 0
    max_height = 0
    max_width_file = None
    max_height_file = None
    for dir in dir_names:
        y_class = dir.split("/")[-1]
        files = glob.glob(os.path.join(dir, "*.jpg"))
        num_files = len(files)
        print "{0}: {1}".format(y_class, num_files)
        total_files += num_files
        for file in files:
            image = imread(file, as_grey=True)
            width, height = image.shape
            total_width += width
            total_height += height
            if width > max_width:
                max_width = width
                max_width_file = file
            if height > max_height:
                max_height = height
                max_height_file = file
    print "total: {}".format(total_files)
    print "largest size: {0}, {1}".format(max_width, max_height)
    print "avg size: {0}, {1}".format(total_width/total_files, total_height/total_files)
    print "max width file: {}".format(max_width_file)
    print "max height file: {}".format(max_height_file)


def load_data(image_size=25):
    dir_names = glob.glob(os.path.join("data", "train", "*"))
    num_rows = len(glob.glob(os.path.join("data", "train", "*", "*.jpg")))
    num_pixels = image_size * image_size
    num_features = num_pixels
    X = np.zeros((num_rows, num_features))
    y = np.zeros((num_rows))

    i = 0
    label = 0
    target_classes = []

    print "Reading images"
    for dir in dir_names:
        current_class = dir.split("/")[-1]
        target_classes.append(current_class)
        files = glob.glob(os.path.join(dir, "*.jpg"))
        for file in files:
            image = imread(file, as_grey=True)
            image = resize(image, (image_size, image_size))
            X[i, 0:num_pixels] = np.reshape(image, (1, num_pixels))
            y[i] = label
            i += 1
            report_prog = [int((j + 1)*num_rows/20.) for j in range(20)]
            if i in report_prog: print np.ceil(i * 100.0/num_rows), "% done"
        label += 1
    return X, y

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


if __name__ == "__main__":
    load_data()