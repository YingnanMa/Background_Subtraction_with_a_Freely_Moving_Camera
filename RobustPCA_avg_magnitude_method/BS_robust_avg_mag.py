import cv2
from itertools import product
import numpy as np
from Robust_PCA import Robust_pca


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
def get_mask(pca_foreground_matrix, prvs):

    mean_value = pca_foreground_matrix.mean()

    mask_matrix = pca_foreground_matrix - mean_value

    mask_matrix = np.absolute(mask_matrix*3)

    mask_matrix = mask_matrix.astype(np.uint8)

    total_number = 0

    matrix_shape = mask_matrix.shape

    count = 0

    for x, y in product(range(matrix_shape[0]), range(matrix_shape[1])):
        if mask_matrix[x][y] < 100:
            total_number += mask_matrix[x][y]
            count += 1

    average_number = total_number / count + 30

    foreground_matrix, binary_mask = easy_thresholding(prvs, mask_matrix, average_number)

    return foreground_matrix, binary_mask


"""
function implement_pca_betweem_two_frames() will implement
Robust PCA for every two frame, combined with avg magnitude method

Parameters:
    image1 - the previous image
    image2 - the new image

"""
def implement_pca_betweem_two_frames(image1, image2):

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
    foreground_matrix, binary_mask = get_mask(pca_background_matrix, prvs)

    #convert to uint8
    pca_foreground_matrix= pca_foreground_matrix.astype(np.uint8)
    pca_background_matrix= pca_background_matrix.astype(np.uint8)
    foreground_matrix = foreground_matrix.astype(np.uint8)
    binary_mask = binary_mask.astype(np.uint8)

    #write image
    cv2.imwrite('pca_back_ground_matrix_'+str(image1)+'.png',pca_background_matrix)
    cv2.imwrite('pca_fore_ground_matrix_'+str(image1)+'.png',pca_foreground_matrix)
    cv2.imwrite('modified_pca_binary_mask_'+str(image1)+'.png',binary_mask)
    cv2.imwrite('modified_pca_true_scene'+str(image1)+'.png',foreground_matrix)

    #destroy table
    cv2.destroyAllWindows()


def main():

    #implement background subtraction to all frames
    pre = "bear02_0"
    for i in range(100, 376):
        implement_pca_betweem_two_frames(pre + str(i) + ".jpg", pre + str(i+1) + ".jpg")

main()
