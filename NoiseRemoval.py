# -*- coding: utf-8 -*-
"""
@author: Himanshu Garg
UBID : 50292195
"""

import cv2
import numpy as np
import time

s = time.time()

UBID = '50292195'; 
np.random.seed(sum([ord(c) for c in UBID]))

def writeImage(name, img):
    path = "output_imgs/" + name
    cv2.imwrite(path,img)
    print("\n****" + name + " saved****")
    
def dilate(img,el):
    pd = np.int32(np.floor(len(el)/2))
    dImg = np.copy(img)
    for indr,row in enumerate(img):
        if indr < pd or indr > len(img) - pd - 1:
            continue
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            if el[pd,pd] == img[indr,indc]:
                dImg[indr-pd:indr+pd+1,indc-pd:indc+pd+1] = 255
    return dImg
        
def erode(img,el):
    pd = np.int32(np.floor(len(el)/2))
    dImg = np.copy(img)
    #display('',dImg)
    for indr,row in enumerate(img):
        if indr < pd or indr > len(img) - pd - 1:
            continue
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            if not np.array_equal(el,img[indr-pd:indr+pd+1,indc-pd:indc+pd+1]):
                dImg[indr,indc] = 0
    return dImg

def doOpening(img,el):
    oimg = erode(img,el)
    oimg = dilate(oimg,el)
    return oimg

def doClosing(img,el):
    cimg = dilate(img,el)
    cimg = erode(cimg,el)
    return cimg
    
def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def denoise1(img,krnl):       
    d1 = doOpening(img,krnl)
    #writeImage("denoise1open.jpg",d1)
    d1 = doClosing(d1,krnl)
    return d1

def denoise2(img,krnl):
    d2 = doClosing(img,krnl)
    #writeImage("denoise2close.jpg",d2)
    d2 = doOpening(d2,krnl)
    return d2

def getBoundary(img1,krnl):
    boundImg = img1 - erode(img1,krnl)
    return boundImg

noiseImg = cv2.imread("original_imgs/noise.jpg",0)
noiseImgB = cv2.threshold(noiseImg,127,255,cv2.THRESH_BINARY)[1]
krnl = np.array([[255,255,255],[255,255,255],[255,255,255]])

dImg1 = denoise1(noiseImgB,krnl)
dImg2 = denoise2(noiseImgB,krnl)

bImg1 = getBoundary(dImg1,krnl)
bImg2 = getBoundary(dImg2,krnl)

writeImage("res_noise1.jpg",dImg1)
writeImage("res_noise2.jpg",dImg2)
writeImage("res_bound1.jpg",bImg1)
writeImage("res_bound2.jpg",bImg2)

print("Elapsed time: ",time.time()-s)






























