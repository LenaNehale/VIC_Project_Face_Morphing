from face_landmark_detection import generate_face_correspondences
from delaunay_triangulation import make_delaunay
from face_morph import generate_mean_image

import subprocess
import argparse
import shutil
import os
import cv2

def doMorphing(imgList):

    [size, imgList, pointsList, list3] = generate_face_correspondences(imgList)

    tri = make_delaunay(size[1], size[0], list3, imgList)

    mean_image=generate_mean_image(imgList,pointsList,tri,size,"output")
    
    return mean_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--imgList", required=True, help="The list of images paths")
    args = parser.parse_args()

    imgList= = cv2.imread([cv2.imread(img) for img in args.imgList])
    
    doMorphing(imgList)
