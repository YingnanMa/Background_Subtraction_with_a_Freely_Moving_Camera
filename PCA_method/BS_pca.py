import cv2
from itertools import product
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

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
function implement_pca_betweem_two_frames() will implement normal PCA for every two frame

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
    _, angle_matrix = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees = True)

    #implement normal PCA based on the coarse foreground
    sklearn_pca=sklearnPCA()
    angle_std=StandardScaler().fit_transform(angle_matrix)
    sklearn_pca.fit_transform(angle_std)

    #convert to uint8
    pca_implement=angle_std.astype(np.uint8)

    #write image
    cv2.imwrite('pca_fore_ground_matrix_'+str(image1)+'.png',pca_implement)

    #destroy table
    cv2.destroyAllWindows()

def main():

    #implement background subtraction to all frames
    pre = "bear02_0"
    for i in range(100, 376):
        implement_pca_betweem_two_frames(pre + str(i) + ".jpg", pre + str(i+1) + ".jpg")

main()
