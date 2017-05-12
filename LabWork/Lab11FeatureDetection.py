import cv2
import numpy as np
from matplotlib import pyplot as plt

from drawMatches import drawMatches



# Load colour image in grayscale
img = cv2.imread('GMIT1.jpg',)
img2 = cv2.imread('GMIT2.jpg',)

orig_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


#cv2.imshow('image',img) 
#cv2.imshow('gray_image',gray_image) 

imgHarris = orig_img.copy()
imgShiTomasi = orig_img.copy()
imgOrb = orig_img.copy()


# imgOut = cv2.GaussianBlur(imgIn,(KernelSizeWidth, KernelSizeHeight),0)
#imgOut = cv2.GaussianBlur(gray_image, (5,5),0)
#imgOut = cv2.GaussianBlur(gray_image, (13,13),0)

# Edge detection using the Sobel Operator. Sobel  Sobel operator detects horizontal and vertical edges by multiplying each pixel by the following two kernels
#sobelHorizontal = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5) # x dir
#sobelVertical = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5) # y dir
blockSize = 2
aperture_size = 3
k = 0.04
dst = cv2.cornerHarris(gray_image, blockSize, aperture_size, k)

threshold = 0.1; #number between 0 and 1
for i in range(len(dst)):
	for j in range(len(dst[i])):
		if dst[i][j] > (threshold*dst.max()):
			cv2.circle(imgHarris,(j,i),3,(255, 0, 0),-1)

maxCorners	= 80		
qualityLevel = 0.01
minDistance = 10
corners = cv2.goodFeaturesToTrack(gray_image,maxCorners,qualityLevel,minDistance)

for i in corners:
	x,y = i.ravel()
	cv2.circle(imgShiTomasi,(x,y),3,(255, 0, 0),-1)

# Initiate ORB-SIFT detector
orb = cv2.ORB(50) #Modify ORB so that it only returns 50 features at a maximum (rather than 500 which is the default)
# find the keypoints and descriptors with ORB-SIFT
kp1, des1 = orb.detectAndCompute(gray_image,None)
kp2, des2 = orb.detectAndCompute(gray_image2,None)

# draw only keypoints location,not size and orientation
imgOrb = cv2.drawKeypoints(imgOrb,kp1,color=(255,0,0))


nrows = 6 # number of rows in window
ncols = 1 # number of columns in window

# Subplotting images in window
plt.subplot(nrows, ncols,1), plt.imshow(orig_img,) 
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2), plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.subplot(nrows, ncols,3), plt.imshow(imgHarris,) 
plt.title('Harris'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,4), plt.imshow(imgShiTomasi,)
plt.title('GFTT'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,5), plt.imshow(imgOrb,) 
plt.title('SIFT/ORB'), plt.xticks([]), plt.yticks([])



plt.show() #calling method
