# reference : https://github.com/geek-ai/1m-agents/blob/master/src/plot_largest_group.py
import numpy as np
import cv2
import os

def make_video(images, outvid=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = cv2.imread(image)

        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


#is_scale function uses thresholding to check if there is scaling operation in video
def is_scale(img1):
    white_points = 0
    img1_shape = img1.shape
    total_pixels = img1_shape[0] * img1_shape[1]
    for i in range(img1_shape[0]):
        for j in range(img1_shape[1]):
            if img1[i][j] > 100:
                white_points += 1
    if white_points / total_pixels > 0.25:
        return True
    else:
        return False



def main():
    images = []

    pre = "modified_pca_binary_mask_bear02_0"

    mag_pre = "mag_pca_binary_mask_bear02_0"

    save_list = []

    #repalce ith frame by using mag_pre
    for i in range(100, 375):
        print(i)

        img1 = cv2.imread(str(pre + str(i) + ".jpg.png"))
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

        if is_scale(img1):
            images.append(str(mag_pre + str(i) + ".jpg.png"))
        else:
            images.append(str(pre + str(i) + ".jpg.png"))

    a = make_video(images, fps = 15, outvid = "RobustPCA_combined_mask.avi")

main()
