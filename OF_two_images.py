import cv2
from itertools import product
import numpy as np


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
function easy_thresholding() will set a thresholding value and filter the angle_matrix

Parameters:
    img - the grayscale image matrix
    angles_matrix - the matrix represneted by pi value.
    thresholding - corresponding pixel will be black if value is greater than thresholding.

Returns:
    foreground_matrix - the foreground image filterd by thresholding value
    binary_mask - the pixel that degree that greater than thresholding will be 255(black)
"""
def easy_thresholding(img, angle_matrix, thresholding):
    
    matrix_shape = angle_matrix.shape
    
    binary_mask = np.zeros(matrix_shape)
    foreground_matrix = np.zeros(matrix_shape)
    
    for x, y in product(range(matrix_shape[0]), range(matrix_shape[1])):
        if angle_matrix[x][y] > thresholding:
            binary_mask[x][y] = 255
            foreground_matrix[x][y] = img[x][y]
            
    return foreground_matrix, binary_mask

    


def main():
    car1 = cv2.imread("car1.jpg")
    car2 = cv2.imread("car2.jpg")
    
    prvs = cv2.cvtColor(car1,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(car2,cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    #cv2.imshow('detect preview', draw_flow(car2, prvs, flow, 16))
    #k = cv2.waitKey(0)
    
    #convert_to_angles(flow)
    
    # angle_matrix is measure by degree now. _ is magnitude
    _, angle_matrix = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees = True)
    easy_foreground_matrix, easy_binary_mask = easy_thresholding(prvs, angle_matrix, 170)
    
    easy_foreground_matrix = easy_foreground_matrix.astype(np.uint8) 
    easy_binary_mask = easy_binary_mask.astype(np.uint8) 
    
    cv2.imshow('easy_binary_mask',easy_binary_mask)
    cv2.imshow('easy_foreground_matrix', easy_foreground_matrix)
    
    cv2.imwrite('easy_binary_mask.png',easy_binary_mask) 
    cv2.imwrite('easy_foreground_matrix.png',easy_foreground_matrix)
    
    k = cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    
main()