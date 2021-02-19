#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import dlib
import numpy as np
import cv2
from sklearn.decomposition import PCA

def keypoints(img, n_kp = 68):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    faces = detector(img, 1)
    
    assert len(faces) > 0, "no face detected"
    assert len(faces) == 1, "many faces detected"
    
    shape = predictor(img, faces[0])
    
    kp = []
    for i in range(n_kp) : 
        kp.append((shape.part(i).x, shape.part(i).y))
        
    #add the image's corners as keypoints
    kp.append((0,0))
    kp.append((0,img.shape[0] - 1))
    kp.append((img.shape[1] - 1,0))
    kp.append((img.shape[1] - 1, img.shape[0] - 1))
    
    return np.asarray(kp, dtype=int)
   
    
# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return np.asarray(dst,dtype=np.uint8)


def morph_triangle(src, dst, t1, t2) :
    # Find bounding rectangle for each triangle
    r1 = (min(t1[0], t1[2], t1[4]), min(t1[1], t1[3], t1[5]), max(t1[0], t1[2], t1[4]), max(t1[1], t1[3], t1[5]))
    r2 = (min(t2[0], t2[2], t2[4]), min(t2[1], t2[3], t2[5]), max(t2[0], t2[2], t2[4]), max(t2[1], t2[3], t2[5]))

    # coords of triangles relatively to their respective bounding boxes
    t1Rect = []
    t2Rect = []

    for i in range(0, 3):
        t1Rect.append(((t1[2*i] - r1[0]),(t1[2*i+1] - r1[1])))
        t2Rect.append(((t2[2*i] - r2[0]),(t2[2*i+1] - r2[1])))
        
    # Get mask 
    mask = np.zeros((r2[3]-r2[1], r2[2]-r2[0], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    srcRect = src[r1[1]:r1[3], r1[0]:r1[2]]

    size = (r2[2]-r2[0], r2[3]-r2[1])
    warpImg = apply_affine_transform(srcRect, t1Rect, t2Rect, size)
    
    # Copy triangular region of the rectangular patch to the output image
    dst[r2[1]:r2[3], r2[0]:r2[2]] = dst[r2[1]:r2[3], r2[0]:r2[2]] + warpImg * mask



def triangulation(img,kp) : 
    #init a 2D Subdivision of the image
    subdiv = cv2.Subdiv2D((0,0,img.shape[1], img.shape[0]))
    for x,y in kp : 
        subdiv.insert((x,y))
    
    #compute triangulation
    triangles =  np.int_(subdiv.getTriangleList())
    
    #find the the indices of the triangles vertices in kp
    inverted_index = {tuple(kp[i]) : i for i in range(len(kp))}
    indices = []
    
    for x0,y0,x1,y1,x2,y2 in triangles : 
        indices.append((inverted_index[(x0,y0)], inverted_index[(x1,y1)], inverted_index[(x2,y2)]))
    
    return triangles,indices


def morph_image(img,kp2):
    #initial keypoints 
    kp = keypoints(img)
    #find vertices for initial triangulation
    srcTri, indices = triangulation(img,kp)
    #compute triangle coordinates with respect to kp2
    dstTri = [(*kp2[i], *kp2[j], *kp2[k]) for i,j,k in indices]
    
    dst = np.zeros_like(img)
    
    #create a blank image and fill it with warped triangles
    for t1, t2 in zip(srcTri, dstTri) : 
        try :
            morph_triangle(img,dst,t1,t2)
        except : 
            pass
        
    return dst


def lip_sync_image(img1, img2, kp=None):
    #find keypoints
    kp1 = kp if kp is not None else keypoints(img1)
    kp2 = keypoints(img2)
    #select only those which describe the mouth
    lip1 = kp1[48:-4]
    lip2 = kp2[48:-4]
    #mean deplacement of the mouth
    translation1 = lip1.sum(axis=0) / len(lip1)
    translation2 = lip2.sum(axis=0) / len(lip2)
    #print(translation1, translation2)
    #compute PCA and normalize lip2 with lip1 PCA
    pca1 = PCA(n_components=2)
    pca1.fit(lip1 - translation1)
    pca2 = PCA(n_components=2)
    pca2.fit(lip2 - translation2)
    #principle components 
    comp1 = pca1.components_[0] if pca1.components_[0] @ [1,0] > 0 else - pca1.components_[0]
    comp2 = pca2.components_[0] if pca2.components_[0] @ [1,0] > 0 else - pca2.components_[0]
    #compute the angle between the main directions of the two mouths
    angle = np.angle((comp2[0] + 1j * comp2[1]) / \
                     (comp1[0] + 1j * comp1[1]), deg=True)
    #compute the scale between mouth 1 and mouth2 (based on range of 1D coord on the pricipals directions)
    coord1D1 = lip1 @ comp1
    coord1D2 = lip2 @ comp2
    scale =  (max(coord1D1) - min(coord1D1)) / (max(coord1D2) - min(coord1D2))
    #print(scale, angle)
    #get the transformatiion matrix
    M = cv2.getRotationMatrix2D((0,0), angle, scale)[:,:2]
    #replace mouth2 keypoints by keypoints from mouth1 but recentered in img2
    kp1[48:-4] = (M @ (lip2 - translation2).T + translation1.reshape((2,1))).T
    #write morph image
    return morph_image(img1,kp1)


def lip_sync(video1, video2, out="out.avi"):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    out = cv2.VideoWriter(out,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    i=0
    while i < 100: 
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2 : 
            out.write(lip_sync_image(frame1, frame2))
            i+= 1
            print(i)
        else : 
            break
    
    cap1.release()
    cap2.release()
    out.release()
    
    
    
def lip_sync_animated_image(img, video2, out="out.avi"):
    cap2 = cv2.VideoCapture(video2)
    img = cv2.imread(img)
    kp = keypoints(img)

    frame_width = img.shape[1]
    frame_height = img.shape[0]
    out = cv2.VideoWriter(out,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    i=0
    while i < 100: 
        ret2, frame2 = cap2.read()
        
        if ret2 : 
            out.write(lip_sync_image(img, frame2, kp))
            i+= 1
            print(i)
        else : 
            break
    
    cap2.release()
    out.release()
    
    

if __name__ == '__main__' :
    vid1 = sys.argv[1]
    vid2 = sys.argv[2]
    out = sys.argv[3]
    
    if vid1[-3:] in ["png", "jpg"] and vid2[-3:] in ["png", "jpg"] : 
        img1, img2 = cv2.imread(vid1), cv2.imread(vid2)
        result = lip_sync_image(img1, img2)
        cv2.imwrite(out, result)
        
    elif vid1[-3:] in ["png", "jpg"] : 
        lip_sync_animated_image(vid1, vid2, out)
        
    else : 
        lip_sync(vid1, vid2, out)
    