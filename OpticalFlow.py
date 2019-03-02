import cv2
import numpy as np
cap = cv2.VideoCapture("video.avi")

# 第一个参数是是否读到了文件，第二个是当前帧
ret, frame1 = cap.read()
# 把这一帧转为夜色空间
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# 给个frame1大小的矩阵
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    # 下一帧
    if ret==False:
        break
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    # 计算optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow.shape)
    # 计算幅度和角度
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next
    


cap.release()
cv2.destroyAllWindows()