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
from SuperPixel import one_image_superpixel as super_pixel
from operator import add


"""
function draw_flow() will visulize the flow

Parameters:
    img - the second image that used to compute the flow
    gray - the first image that used to compute the flow
    flow - the flow matrix represented by offset of x and y

Returns:
    vis - the image matrix

Reference:https://www.digifie.jp/blog/archives/1448
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
def get_mask_ang(pca_background_matrix, prvs, abso__ang_thre, count_ang_thre):

    mean_value = pca_background_matrix.mean()

    mask_matrix = pca_background_matrix - mean_value

    mask_matrix = np.absolute(mask_matrix*abso__ang_thre)

    mask_matrix = mask_matrix.astype(np.uint8)

    total_number = 0

    matrix_shape = mask_matrix.shape

    count = 0

    for x, y in product(range(matrix_shape[0]), range(matrix_shape[1])):
        if mask_matrix[x][y] < 100:
            total_number += mask_matrix[x][y]
            count += 1

    average_number = total_number / count+count_ang_thre

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
def implement_pca_betweem_two_frames_ang(image1, image2, abso__ang_thre, count_ang_thre):

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

    #modify all 360 degree as 0
    angle_shape = angle_matrix.shape
    for i in range(angle_shape[0]):
        for j in range(angle_shape[1]):
            if angle_matrix[i][j] >=359 and angle_matrix[i][j]<=360:
                angle_matrix[i][j]=0

    #implement Robust PCA based on the coarse foreground
    pca_implement=Robust_pca(angle_matrix)
    pca_background_matrix, pca_foreground_matrix=pca_implement.generate_pca()

    #get binary mask
    foreground_matrix, binary_mask = get_mask_ang(pca_background_matrix, prvs,  abso__ang_thre, count_ang_thre)

    #convert to uint8
    pca_foreground_matrix= pca_foreground_matrix.astype(np.uint8)
    pca_background_matrix= pca_background_matrix.astype(np.uint8)
    foreground_matrix = foreground_matrix.astype(np.uint8)
    binary_mask = binary_mask.astype(np.uint8)

    #write image
    cv2.imwrite('ang_pca_binary_mask_'+str(image1)+'.png',binary_mask)

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
def get_mask_mag(pca_background_matrix, prvs,abso__mag_thre, count_mag_thre):

    mean_value = pca_background_matrix.mean()

    mask_matrix = pca_background_matrix - mean_value

    mask_matrix = np.absolute(mask_matrix*abso__mag_thre)

    mask_matrix = mask_matrix.astype(np.uint8)

    total_number = 0

    matrix_shape = mask_matrix.shape

    count = 0

    for x, y in product(range(matrix_shape[0]), range(matrix_shape[1])):
        if mask_matrix[x][y] < 100:
            total_number += mask_matrix[x][y]
            count += 1

    average_number = total_number / count + count_mag_thre

    foreground_matrix, binary_mask = easy_thresholding(prvs, mask_matrix, average_number)

    return foreground_matrix, binary_mask

"""
function implement_pca_betweem_two_frames_mag() will implement
Robust PCA for every two frame, combined with avg magnitude method

Parameters:
    image1 - the previous image
    image2 - the new image

"""
def implement_pca_betweem_two_frames_mag(image1, image2,abso__mag_thre, count_mag_thre):

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
    foreground_matrix, binary_mask = get_mask_mag(pca_background_matrix, prvs,abso__mag_thre, count_mag_thre)

    #convert to uint8
    pca_foreground_matrix= pca_foreground_matrix.astype(np.uint8)
    pca_background_matrix= pca_background_matrix.astype(np.uint8)
    foreground_matrix = foreground_matrix.astype(np.uint8)
    binary_mask = binary_mask.astype(np.uint8)

    #write image
    cv2.imwrite('mag_pca_binary_mask_'+str(image1)+'.png',binary_mask)

    #destroy table
    cv2.destroyAllWindows()


def main():
    #create thresholding dictionary, which tells the thresholding of datasets
    #index0:angle_absolute__thresholding
    #index1:angle_count_thresholding
    #index2:magnitude_absolute__thresholding
    #index3:magnitude_count_thresholding
    #index4:mask_revising_thresholding1
    #index5:mask_revising_thresholding2
    #index6:superpixel_thresholding1
    #index7:superpixel_thresholding2
    thre_dictionary={
    "bear02":[3.7,48,20,10,0.3,0.06,0.33,3]
    }

    #create ground truth dictionary, which tells the frame number of ground truth
    gt_dictionary={
    "bear02":[1,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,458]
    }

    #choose the dataset
    dataset_request=input("Datasets:\n bear02\nChoose the dataset from above : ")

    #check dataset exist
    exist=False
    while not exist:
        if dataset_request not in thre_dictionary.keys():
            dataset_request=input("Dataset does not exist\nDatasets:\n bear02\nChoose the dataset from above : ")
            exist=False
        else:
            exist=True

    #pick target dataset thersholding and ground truth frames
    thre_pick=thre_dictionary[dataset_request]
    gt_pick=gt_dictionary[dataset_request]

    #implement background subtraction to all frames
    pre = dataset_request+"_0"
    for i in gt_pick:

        #check the frames that use avg angle method
        print("angle "+ str(i) + " is processing ")
        if i==458:
           implement_pca_betweem_two_frames_ang(pre + str(i) + ".jpg", pre + str(i-1) + ".jpg",thre_pick[0],thre_pick[1])
        else:
           implement_pca_betweem_two_frames_ang(pre + str(i) + ".jpg", pre + str(i+1) + ".jpg",thre_pick[0],thre_pick[1])
        img_head = "ang_pca_binary_mask_"+dataset_request+"_0"


        #check the frames that use avg magnitude method
        print("magnitude "+ str(i) + " is processing ")
        if i==458:
           implement_pca_betweem_two_frames_mag(pre + str(i) + ".jpg", pre + str(i-1) + ".jpg",thre_pick[2],thre_pick[3])
        else:
           implement_pca_betweem_two_frames_mag(pre + str(i) + ".jpg", pre + str(i+1) + ".jpg",thre_pick[2],thre_pick[3])
        img_head2 = "mag_pca_binary_mask_"+dataset_request+"_0"


        #combine angle result and magnitude result
        mask_ang=cv2.imread(str(img_head + str(i) + ".jpg.png"))
        mask_ang=cv2.cvtColor(mask_ang,cv2.COLOR_BGR2GRAY)
        white_ang=is_scale(mask_ang)
        print("angle "+ str(i) + "  white rate : "+ str(white_ang))
        mask_mag=cv2.imread(str(img_head2 + str(i) + ".jpg.png"))
        mask_mag=cv2.cvtColor(mask_mag,cv2.COLOR_BGR2GRAY)
        white_mag=is_scale(mask_mag)
        print("magnitude "+ str(i) + "  white rate : "+ str(white_mag))
        matrix_shape = mask_mag.shape
        array_length = matrix_shape[0] * matrix_shape[1]
        mask_mag = np.reshape(mask_mag, array_length)
        mask_ang = np.reshape(mask_ang, array_length)
        mask_mag= mask_mag.astype(np.uint16)
        mask_ang= mask_ang.astype(np.uint16)
        combine_mask = mask_mag + mask_ang

        #set adaptive threshodling to implement subtraction revising
        if white_ang>=thre_pick[4]:
            new_mask =[]
            for j in combine_mask:
                if j == 510:
                    new_mask.append(255)
                else:
                    new_mask.append(0)
            new_mask=np.asarray(new_mask)
            new_mask= new_mask.astype(np.uint8)
            cv2.imwrite("modified_pca_binary_mask_"+dataset_request+"_0"+ str(i) + ".jpg.png",\
                        np.reshape(new_mask, matrix_shape))

        #set adaptive threshodling to implement summation revising
        elif white_ang<=thre_pick[5]:
            new_mask = []
            for j in combine_mask:
                if j >= 255:
                    new_mask.append(255)
                else:
                    new_mask.append(0)
            new_mask=np.asarray(new_mask)
            new_mask= new_mask.astype(np.uint8)
            cv2.imwrite("modified_pca_binary_mask_"+dataset_request+"_0"+ str(i) + ".jpg.png",\
                        np.reshape(new_mask, matrix_shape))

        else:
            absname = "ang_pca_binary_mask_"+dataset_request+"_0"+ str(i) + ".jpg.png"
            newname = "modified_pca_binary_mask_"+dataset_request+"_0"+ str(i) + ".jpg.png"
            os.rename(absname, newname)

        #optimized by superpixel
        original_name=pre + str(i) + ".jpg"
        optimized_mask=super_pixel(original_name,"modified_pca_binary_mask_"+dataset_request+"_0"+ str(i) + ".jpg.png",thre_pick[6],thre_pick[7])
        cv2.imwrite( "optimized_mask"+ str(i)+".jpg.png", optimized_mask)
        print("superpixel "+ str(i) + " finished")

main()
