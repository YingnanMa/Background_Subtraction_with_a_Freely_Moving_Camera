import numpy as np
import cv2
np.set_printoptions(threshold=np.inf)
import skimage
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

"""
parameters:
orignal_image: matrix, orignal imgage
mask_imgae: matrix, mask image
number_of_segments int, number of the segments
thresholding float between 0 to 1, thresholding to select segmentation

return: 1-d array, the matrix after superpixel
"""

def find_one_super_pixel(ori_img, mask_img ,number_of_segments, thresholding):
    img_shape = ori_img.shape

    segments_slic = slic(ori_img, number_of_segments, compactness=10, sigma=1)

    segments_slic = np.reshape(segments_slic, (img_shape[0]*img_shape[1]))
    mask_img = np.reshape(mask_img, (img_shape[0]*img_shape[1]))

    number_of_segment = len(np.unique(segments_slic))

    hash_table = np.zeros(number_of_segment)
    mask_table = np.zeros(number_of_segment)
    # 0 black, 1 white

    segments_len = len(segments_slic)

    for i in range(segments_len):
        index = segments_slic[i]
        hash_table[index] += 1
        if mask_img[i] == 1.0:
            mask_table [index] += 1

    ratio_table = mask_table / hash_table

    useful_label = []

    for i in range(len(ratio_table)):
        if ratio_table[i] > thresholding:
            useful_label.append(i)

    new_img = []

    for i in range(segments_len):
        if segments_slic[i] in useful_label:
            new_img.append(1.0)
        else:
            new_img.append(0.0)

    return new_img

"""
parameters:
original_image: str, path for orignal image
mask_image, str, path ofr mask image
"""

def one_image_superpixel(original_image, mask_image,sp_thre1,sp_thre2):

    ori_img = skimage.io.imread(original_image)
    ori_img = img_as_float(ori_img)

    mask_img = skimage.io.imread(mask_image)
    mask_img = img_as_float(mask_img)

    candidates = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    img_shape = ori_img.shape
    array_length = img_shape[0] * img_shape[1]
    sum_of_super_pixel = np.zeros(array_length)

    for i in candidates:
        print("segmentation for ", i)
        super_pixel_result = find_one_super_pixel(ori_img, mask_img, i, sp_thre1)
        for j in range(array_length):
            if super_pixel_result[j] == 1.0:
                sum_of_super_pixel[j] += 1

    average_superpixel_result = np.zeros(array_length)
    for i in range(array_length):
        if sum_of_super_pixel[i] >= sp_thre2:
            average_superpixel_result[i] = 255

    average_superpixel_result=np.reshape(average_superpixel_result, img_shape[:-1])
    return average_superpixel_result
