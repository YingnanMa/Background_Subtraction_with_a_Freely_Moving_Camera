import cv2
import numpy as np


# reference:
# https://www.digifie.jp/blog/archives/1448


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



def main():
    car1 = cv2.imread("car1.jpg")
    car2 = cv2.imread("car2.jpg")
    
    prvs = cv2.cvtColor(car1,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(car2,cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    #hsv = np.zeros_like(car1)
    #hsv[...,1] = 255
    #hsv[...,0] = ang*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    #cv2.imshow('car2',rgb)
    
    cv2.imshow('detect preview', draw_flow(car2, prvs, flow, 16))
    
    
    k = cv2.waitKey(0)
    print(flow)
    
main()