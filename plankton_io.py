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
    for dir_name in dir_names:
        y_class = dir_name.split("/")[-1]
        filenames = glob.glob(os.path.join(dir_name, "*.jpg"))
        num_files = len(filenames)
        print "{0}: {1}".format(y_class, num_files)
        total_files += num_files
        for filename in filenames:
            image = imread(filename, as_grey=True)
            width, height = image.shape
            total_width += width
            total_height += height
            if width > max_width:
                max_width = width
                max_width_file = filename
            if height > max_height:
                max_height = height
                max_height_file = filename
    print "total: {}".format(total_files)
    print "largest size: {0}, {1}".format(max_width, max_height)
    print "avg size: {0}, {1}".format(total_width/total_files, total_height/total_files)
    print "max width file: {}".format(max_width_file)
    print "max height file: {}".format(max_height_file)


def load_data(image_size=25):
    """
    iterates through the folder structure, adding pixels and labels to X and y numpy arrays
    :param image_size: image size after resize
    :return: two numpy arrays, X containing pixels and maybe other stuff, y contains labels
    """
    dir_names = glob.glob(os.path.join("data", "train", "*"))
    num_rows = len(glob.glob(os.path.join("data", "train", "*", "*.jpg")))
    num_pixels = image_size * image_size
    num_features = num_pixels + 4
    X = np.zeros((num_rows, num_features))
    y = np.zeros((num_rows))

    i = 0
    label = 0
    target_classes = []

    print "Reading images"
    for dir_name in dir_names:
        current_class = dir_name.split("/")[-1]
        target_classes.append(current_class)
        filenames = glob.glob(os.path.join(dir_name, "*.jpg"))
        for filename in filenames:
            image = imread(filename, as_grey=True)
            image = resize(image, (image_size, image_size))
            X[i, 0:num_pixels] = np.reshape(image, (1, num_pixels))
            segment_stats = get_largest_segment_summary_stats(image)
            X[i, -4:] = np.array(segment_stats)
            y[i] = label
            i += 1
            report_prog = [int((j + 1)*num_rows/20.) for j in range(20)]
            if i in report_prog:
                print np.ceil(i * 100.0/num_rows), "% done"
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


def dilate_image(imthr, n=4):
    """
    dilate an image by setting each pixel to the max val within an nxn filter
    :param imthr: image in the form of 2d numpy array, should be thresholded
    :return: dilated image in the form of 2d numpy array
    """
    imdilated = morphology.dilation(imthr, np.ones((n, n)))
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


def get_largest_segment(labeled_image):
    """
    returns the largest segment of a labelled image using regionprop.filled_area
    :param labeled_image: 2d numpy array containing segment labels of each pixel
    :return: regionprop object that contains the largest filled_area
    """
    regions = measure.regionprops(labeled_image)
    largest_region = None
    for region in regions:
        if largest_region is None:
            largest_region = region
        elif region.filled_area > largest_region.filled_area:
            largest_region = region
    return largest_region


def get_largest_segment_summary_stats(image):
    """
    gets filled_area, perimeter, major_axis_length, and minor_axis_length of a labeled image's largest segment
    :param labeled_image: 2d numpy array containing segment labels of each pixel in an image
    :return: a list of filled_area(int), perimeter(float), and major_ and minor_axis_length (float)
    """
    labeled_image = segment_image(image)
    largest_region = get_largest_segment(labeled_image)
    if largest_region:
        return [largest_region.filled_area, largest_region.perimeter, largest_region.major_axis_length,
                largest_region.minor_axis_length]
    else:
        return 0, 0, 0, 0


if __name__ == "__main__":
    X, y = load_data()
