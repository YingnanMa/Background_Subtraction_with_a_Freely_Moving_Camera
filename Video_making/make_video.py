# reference : https://github.com/geek-ai/1m-agents/blob/master/src/plot_largest_group.py

import cv2
import os
def make_video(images, outvid=None, fps=5, size=None,
               is_color=True, format="XVID",ipath=None):
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

        img = cv2.imread(ipath+image)

        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def make_one_video(path, video_name):
    images = os.listdir(path)
    make_video(images, video_name, fps=15, ipath = path)

def main():
    folder_names = ['Origin', 'PCA', 'RPCA_ANG', 'RPCA_Replace', 'RPCA_revised','RPCA_SP']
    for i in folder_names:
        make_one_video(i + "/" , i + ".mp4")
main()