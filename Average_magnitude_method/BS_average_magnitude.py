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

def implement_average_magnitude_betweem_two_frames(image1, image2):
    #read image
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    #convert BGR to Gray
    prvs = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    #calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #obtain angle matrix: _ is magnitude and angle_matrix is measure by degree now.
    mag, angle_matrix = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees = True)

    #get average angle matrix
    average = mag.mean()

    #get foreground matrix by subtraction
    foreground = mag - average

    #get absolute value of foreground matrix
    foreground = np.absolute(foreground)

    #convert to uint8
    foreground = foreground.astype(np.uint8)

    #show image
    #cv2.imshow('foreground', foreground)

    #write image
    cv2.imwrite('foreground_'+str(image1)+'.png',foreground*50)

    #destroy table
    cv2.destroyAllWindows()
    
    
def main():
    
    pre = "bear02_0"
    for i in range(100, 450):
        implement_average_magnitude_betweem_two_frames(pre + str(i) + ".jpg", pre + str(i+1) + ".jpg")        
    
main()
