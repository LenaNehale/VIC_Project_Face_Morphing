import numpy as np
import cv2
import sys
import os
import math
from subprocess import Popen, PIPE
from PIL import Image
from google.colab.patches import cv2_imshow

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size) :
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha-blends triangular regions one by one (with alpha=1/n), from  all the n images in imgList to the image img :
def morph_triangle(imgList, img, tList, warp_t) :
    '''
    Arguments : 
        imgList : the list of input images
        img : the target image which will contain the mean morphing of the input images
        tList : the list of Delaunay triangles for each of the n images. Shape : [n, number of triangles, (3,2)] (each triangle is defined by 3 points with 2 coordinates x and y )
        warp_t : the target Delaunay triangle points. Shape [3, 2] (each triangle is defined by 3 points with 2 coordinates x and y)
        Remark : points range from 1 to 76 as there are 76 keypoints detected  (see face_landmark_detection file).
        
    Returns : 
        It modifies the pixels inside the triangle warp_t of the image img : these pixels are obtained by by warping the n correponding Delaunay triangles in the n images
        
    '''
    n_input_images=len(imgList)
    
    # Find bounding rectangle for each triangle
    rList = [cv2.boundingRect(np.float32([t])) for t in tList]
    warp_r = cv2.boundingRect(np.float32([warp_t]))

    # Offset points by left top corner of the respective rectangles
    tRectList = [[] for i in range(n_input_images)]
    
    warp_tRect = []

    for i in range(0, 3):
        warp_tRect.append(((warp_t[i][0] - warp_r[0]),(warp_t[i][1] - warp_r[1])))
        for j,(t,r) in enumerate(zip(tList,rList)):
            tRectList[j].append(((t[i][0] - r[0]),(t[i][1] - r[1])))

    # Get mask by filling triangle
    mask = np.zeros((warp_r[3], warp_r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(warp_tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    imgRectList = [img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] for (img,r) in zip(imgList,rList)]

    size = (warp_r[2], warp_r[3])
    warpImageList = [apply_affine_transform(imgRect, tRect, warp_tRect, size) for (imgRect,tRect) in zip(imgRectList, tRectList)]

    # Alpha blend rectangular patches
    imgRect = sum([1/len(imgList)*warpImage for warpImage in warpImageList ])
    #print(imgRect.shape)
    #print(mask.shape)
    
    # Copy triangular region of the rectangular patch to the output image
    img[warp_r[1]:warp_r[1]+warp_r[3], warp_r[0]:warp_r[0]+warp_r[2]] = img[warp_r[1]:warp_r[1]+warp_r[3], warp_r[0]:warp_r[0]+warp_r[2]] * ( 1 - mask ) + imgRect * mask

#Generates the mean image of n images
def generate_mean_image(imgList,pointsList,tri_list,size):
    '''
    Arguments : 
        imgList : the list of input images
        pointsList : a list of size n (number of input images), where each sublist contains the coordinates (x,y) of the face keypoints
        tri_list : A list of common the Delaunay triangulation for the n images (The triangulation is done with the same keypoint refernce (reference going from 1 to 76, as for each image there are 76 keypoints) in order to be able to perform morphing, but the keypoint coordinates differ from one image to another ! )
        size : the size of our images.
    Returns:
    The mean morphed image.
        
    
    '''
    n_input_images=len(imgList)
    

    # Convert Mat to float data type
    imgList=[ np.float32(img) for img in imgList]

    # Read array of corresponding points
    points = []
  
    # Compute weighted average point coordinates
    for i in range(0, len(pointsList[0])):
        x = sum([1/n_input_images*points[i][0] for points in pointsList])
        y = sum([1/n_input_images*points[i][1] for points in pointsList])
        points.append((x,y))
    
    # Allocate space for final output
    morphed_frame = np.zeros(imgList[0].shape, dtype = imgList[0].dtype)

    for i in range(len(tri_list)):    
        x = int(tri_list[i][0])
        y = int(tri_list[i][1])
        z = int(tri_list[i][2])
        
        tList = [[points[x], points[y], points[z]] for points in pointsList]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morph_triangle(imgList, morphed_frame, tList, t)
        
        pt1 = (int(t[0][0]), int(t[0][1]))
        pt2 = (int(t[1][0]), int(t[1][1]))
        pt3 = (int(t[2][0]), int(t[2][1]))

        #cv2.line(morphed_frame, pt1, pt2, (255, 255, 255), 1, 8, 0)
        #cv2.line(morphed_frame, pt2, pt3, (255, 255, 255), 1, 8, 0)
        #cv2.line(morphed_frame, pt3, pt1, (255, 255, 255), 1, 8, 0)

    #res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
    #res.save(p.stdin,'JPEG')
    #cv2_imshow(res)
    cv2_imshow(morphed_frame)
    return morphed_frame



'''
EXECUTING THE FILES : 



'''