import cv2
from itertools import product
import numpy as np
from Robust_PCA import Robust_pca
import os
import skimage
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# reference:
# https://www.digifie.jp/blog/archives/1448

"""
function draw_flow() will visulize the flow

Parameters:
    img - the second image that used to compute the flow
    gray - the first image that used to compute the flow
    flow - the flow matrix represented by offset of x and y

Returns:
    vis - the image matrix
"""

def draw_flow(img, gray, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis = 255 - vis
    rad = int(step/2)

    i = 0
    for (x1, y1), (x2, y2) in lines:
        pv = img[y1, x1]
        col = (int(pv[0]), int(pv[1]), int(pv[2]))
        r = rad - int(rad * abs(fx[i]) *.05)
        cv2.circle(vis, (x1, y1), abs(r), col, -1)
        i+=1
    cv2.polylines(vis, lines, False, (255, 255, 0))
    return vis


"""
function easy_thresholding() will set a thresholding value and filter the angle_matrix

Parameters:
    img - the grayscale image matrix
    angles_matrix - the matrix represneted by pi value.
    thresholding - corresponding pixel will be black if value is greater than thresholding.

Returns:
    foreground_matrix - the foreground image filterd by thresholding value
    binary_mask - the pixel that degree that greater than thresholding will be 255(black)
"""
def easy_thresholding(img, given_matrix, thresholding):

    matrix_shape = given_matrix.shape

    binary_mask = np.zeros(matrix_shape)
    foreground_matrix = np.zeros(matrix_shape)

    for x, y in product(range(matrix_shape[0]), range(matrix_shape[1])):
        if given_matrix[x][y] > thresholding:
            binary_mask[x][y] = 255
            foreground_matrix[x][y] = img[x][y]

    return foreground_matrix, binary_mask


"""
function get_mask() will get binary mask, using foreground matrix obtained from robust PCA
Parameters:
    pca_foreground_matrix - foreground matrix obtained from robust PCA
    prvs - the previous image

"""
def get_mask(pca_background_matrix, prvs):

    mean_value = pca_background_matrix.mean()

    mask_matrix = pca_background_matrix - mean_value

    mask_matrix = np.absolute(mask_matrix*3.7)

    mask_matrix = mask_matrix.astype(np.uint8)

    total_number = 0

    matrix_shape = mask_matrix.shape

    count = 0

    for x, y in product(range(matrix_shape[0]), range(matrix_shape[1])):
        if mask_matrix[x][y] < 100:
            total_number += mask_matrix[x][y]
            count += 1

    average_number = total_number / count+48

    foreground_matrix, binary_mask = easy_thresholding(prvs, mask_matrix, average_number)

    return foreground_matrix, binary_mask

"""
function convert_to_angles() will transfer flow matrix to angle matrix, which
represented by pi value.
Parameters:
    flow_matrix - the flow matrix represented by offset of x and y
Returns:
    angles_matrix - the matrix represneted by degree value.
"""
def convert_to_angles(flow_matrix):

    flow_matrix_shape = flow_matrix.shape
    angles_matrix = np.zeros(flow_matrix_shape[:2])

    for x, y in product(range(flow_matrix_shape[0]), range(flow_matrix_shape[1])):
        offset_data = flow_matrix[x][y]
        angle = np.arctan(offset_data[0]/offset_data[1])
        angles_matrix[x][y] = angle

    return angles_matrix

"""
function implement_pca_betweem_two_frames_ang() will implement
Robust PCA for every two frame, combined with avg angle method

Parameters:
    image1 - the previous image
    image2 - the new image

"""
def implement_pca_betweem_two_frames_ang(image1, image2):

    #read image
    pic1 = cv2.imread(image1)
    pic2 = cv2.imread(image2)

    #convert BGR to Gray
    prvs = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(pic2,cv2.COLOR_BGR2GRAY)

    #calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #obtain angle matrix: _ is magnitude and angle_matrix is measure by degree now.
    _, angle_matrix = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees = True)
    #angle_matrix=convert_to_angles(flow)

    #modify all 360 degree as 0
    angle_shape = angle_matrix.shape
    for i in range(angle_shape[0]):
        for j in range(angle_shape[1]):
            #if angle_matrix[i][j] ==360:
            if angle_matrix[i][j] >=359 and angle_matrix[i][j]<=360:
                angle_matrix[i][j]=0

    #print(angle_matrix)
    #implement Robust PCA based on the coarse foreground
    pca_implement=Robust_pca(angle_matrix)
    pca_background_matrix, pca_foreground_matrix=pca_implement.generate_pca()

    #get binary mask
    foreground_matrix, binary_mask = get_mask(pca_background_matrix, prvs)

    #convert to uint8
    pca_foreground_matrix= pca_foreground_matrix.astype(np.uint8)
    pca_background_matrix= pca_background_matrix.astype(np.uint8)
    foreground_matrix = foreground_matrix.astype(np.uint8)
    binary_mask = binary_mask.astype(np.uint8)

    #write image
    #cv2.imwrite('pca_back_ground_matrix_'+str(image1)+'.png',pca_background_matrix)
    #cv2.imwrite('pca_fore_ground_matrix_'+str(image1)+'.png',pca_foreground_matrix)
    cv2.imwrite('ang_pca_binary_mask_'+str(image1)+'.png',binary_mask)
    #cv2.imwrite('modified_pca_true_scene'+str(image1)+'.png',foreground_matrix)

    #destroy table
    cv2.destroyAllWindows()


#is_scale function uses thresholding to check white rate
def is_scale(img1):
    white_points = 0
    img1_shape = img1.shape
    total_pixels = img1_shape[0] * img1_shape[1]
    for i in range(img1_shape[0]):
        for j in range(img1_shape[1]):
            if img1[i][j] > 100:
                white_points += 1
    white_rate= white_points / total_pixels
    return white_rate


"""
function get_mask_mag() will get binary mask for mag method, using foreground matrix obtained from robust PCA
Parameters:
    pca_foreground_matrix - foreground matrix obtained from robust PCA
    prvs - the previous image

"""
def get_mask_mag(pca_foreground_matrix, prvs):

    mean_value = pca_foreground_matrix.mean()

    mask_matrix = pca_foreground_matrix - mean_value

    mask_matrix = np.absolute(mask_matrix*20)

    mask_matrix = mask_matrix.astype(np.uint8)

    total_number = 0

    matrix_shape = mask_matrix.shape

    count = 0

    for x, y in product(range(matrix_shape[0]), range(matrix_shape[1])):
        if mask_matrix[x][y] < 100:
            total_number += mask_matrix[x][y]
            count += 1

    average_number = total_number / count + 10

    foreground_matrix, binary_mask = easy_thresholding(prvs, mask_matrix, average_number)

    return foreground_matrix, binary_mask

"""
function implement_pca_betweem_two_frames_mag() will implement
Robust PCA for every two frame, combined with avg magnitude method

Parameters:
    image1 - the previous image
    image2 - the new image

"""
def implement_pca_betweem_two_frames_mag(image1, image2):

    #read image
    pic1 = cv2.imread(image1)
    pic2 = cv2.imread(image2)

    #convert BGR to Gray
    prvs = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(pic2,cv2.COLOR_BGR2GRAY)

    #calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #obtain angle matrix: _ is magnitude and angle_matrix is measure by degree now.
    mag, angle_matrix = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees = True)

    #implement Robust PCA based on the coarse foreground
    pca_implement=Robust_pca(mag)
    pca_background_matrix, pca_foreground_matrix=pca_implement.generate_pca()

    #get binary mask
    foreground_matrix, binary_mask = get_mask_mag(pca_background_matrix, prvs)

    #convert to uint8
    pca_foreground_matrix= pca_foreground_matrix.astype(np.uint8)
    pca_background_matrix= pca_background_matrix.astype(np.uint8)
    foreground_matrix = foreground_matrix.astype(np.uint8)
    binary_mask = binary_mask.astype(np.uint8)

    #write image
    #cv2.imwrite('pca_back_ground_matrix_'+str(image1)+'.png',pca_background_matrix)
    #cv2.imwrite('pca_fore_ground_matrix_'+str(image1)+'.png',pca_foreground_matrix)
    cv2.imwrite('mag_pca_binary_mask_'+str(image1)+'.png',binary_mask)
    #cv2.imwrite('mag_pca_true_scene'+str(image1)+'.png',foreground_matrix)

    #destroy table
    cv2.destroyAllWindows()


def main():

    #implement background subtraction to all frames using avg angle method
    pre = "bear02_0"
    for i in range(100,375):
        implement_pca_betweem_two_frames_ang(pre + str(i) + ".jpg", pre + str(i+1) + ".jpg")

        #check the frames that use avg angle method
        img_head = "ang_pca_binary_mask_bear02_0"
        img_check = cv2.imread(str(img_head + str(i) + ".jpg.png"))
        img_check = cv2.cvtColor(img_check,cv2.COLOR_BGR2GRAY)
        white_ang=is_scale(img_check)
        print(white_ang)

        #check the frames that use avg magnitude method
        implement_pca_betweem_two_frames_mag(pre + str(i) + ".jpg", pre + str(i+1) + ".jpg")
        img_head2 = "mag_pca_binary_mask_bear02_0"
        img_check2 = cv2.imread(str(img_head2 + str(i) + ".jpg.png"))
        img_check2 = cv2.cvtColor(img_check2,cv2.COLOR_BGR2GRAY)
        white_mag=is_scale(img_check2)
        print(white_mag)

        #set threshodling to choose angle method or magnitude method
        if white_ang>0.19:
            if white_ang>white_mag:
                absname = "mag_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
                newname = "modified_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
                os.rename(absname, newname)
            else:
                absname = "ang_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
                newname = "modified_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
                os.rename(absname, newname)
        elif white_ang<0.06:
            if white_ang>white_mag:
                absname = "ang_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
                newname = "modified_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
                os.rename(absname, newname)
            else:
                absname = "mag_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
                newname = "modified_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
                os.rename(absname, newname)
        else:
            absname = "ang_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
            newname = "modified_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png"
            os.rename(absname, newname)

        #implement SLIC superpixel
        img_compare=cv2.imread("modified_pca_binary_mask_bear02_0"+ str(i) + ".jpg.png")
        img_compare=cv2.cvtColor(img_compare,cv2.COLOR_BGR2GRAY)
        white_compare=is_scale(img_compare)

        #compare superpixel with foreground result
        img_super=skimage.io.imread(pre+str(i)+".jpg")
        img_super=img_as_float(img_super)
        img_shape=img_compare.shape
        optimized_mask=np.zeros(img_shape)

        #handle edge case
        if white_compare<0.09:
            n_segments=128
            thresh_super=0.2
        #handle good case
        elif white_compare<0.12:
            n_segments=256
            thresh_super=0.25
        #handle detailed case
        elif white_compare<0.16:
            n_segments=640
            thresh_super=0.3
        #handle noise case
        else:
            n_segments=1024
            thresh_super=0.4

        #implement slic superpixel
        segments_slic = slic(img_super, n_segments, compactness=10, sigma=1)
        number_of_segment = len(np.unique(segments_slic))
        for s in range(number_of_segment):
            total_number = 0
            white_count = 0
            position = []
            for j in range(img_shape[0]):
                for k in range(img_shape[1]):
                    if segments_slic[j][k] == s:
                        total_number +=1
                        position.append([j, k])
                        if img_compare[j][k] > 100:
                            white_count += 1
            print(s)
            if white_count / total_number > thresh_super:
                for position_index in position:
                    optimized_mask[position_index[0]][position_index[1]] = 255

        cv2.imwrite( "optimized_mask"+ str(i)+".jpg.png", optimized_mask)

main()
