import sys
import os
import dlib
import glob
import numpy as np
from skimage import io
import cv2
from imutils import face_utils

class NoFaceFound(Exception):
   """Raised when there is no face found"""
   pass




def generate_face_correspondences(imgList):
    '''
    Arguments:
        imgList : a list of  cv2.read() face images (All the images must have the same shape ! ) , of len n
    Returns:
        size : the size of our images (m,p,num_of_channels)
        imgList : the list of our initial images
        lists : a list of size n, where each sublist contains the coordinates (x,y) of the face keypoints, found using dlib library
        narray : contains the means of each keypoint across the n images, as well as background points for a better morphing.
        
        
        
    '''
    
    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/content/Face-Morphing/code/utils/shape_predictor_68_face_landmarks.dat')
    corresp = np.zeros((68,2))

    
    lists=[]
   

    for img in imgList:

        size = (img.shape[0],img.shape[1])

        currList =[]

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        dets = detector(img, 1)

        try:
            if len(dets) == 0:
                raise NoFaceFound
        except NoFaceFound:
            print("Sorry, but I couldn't find a face in the image.")

        

        for k, rect in enumerate(dets):
            
            # Get the landmarks/parts for the face in rect.
            shape = predictor(img, rect)
            '''
            plot_shape = face_utils.shape_to_np(shape)
            for (x, y) in plot_shape:
	              cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2_imshow(img)
            '''
            # corresp = face_utils.shape_to_np(shape)
            
            for i in range(0,68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y
                # cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

            # Add back the background
            currList.append((1,1))
            currList.append((size[1]-1,1))
            currList.append(((size[1]-1)//2,1))
            currList.append((1,size[0]-1))
            currList.append((1,(size[0]-1)//2))
            currList.append(((size[1]-1)//2,size[0]-1))
            currList.append((size[1]-1,size[0]-1))
            currList.append(((size[1]-1),(size[0]-1)//2))
        lists.append(currList)

    # Add back the background
    narray = corresp/len(imgList)
    narray = np.append(narray,[[1,1]],axis=0)
    narray = np.append(narray,[[size[1]-1,1]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,1]],axis=0)
    narray = np.append(narray,[[1,size[0]-1]],axis=0)
    narray = np.append(narray,[[1,(size[0]-1)//2]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,size[0]-1]],axis=0)
    narray = np.append(narray,[[size[1]-1,size[0]-1]],axis=0)
    narray = np.append(narray,[[(size[1]-1),(size[0]-1)//2]],axis=0)
    
    return [size,imgList,lists,narray]