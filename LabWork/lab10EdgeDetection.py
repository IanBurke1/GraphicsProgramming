import cv2
import numpy as np
from matplotlib import pyplot as plt



# Load colour image in grayscale
#img = cv2.imread('GMIT.jpg',)
img = cv2.imread('eyreSquare.jpg',)

orig_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('image',img) 
#cv2.imshow('gray_image',gray_image) 

# imgOut = cv2.GaussianBlur(imgIn,(KernelSizeWidth, KernelSizeHeight),0)
imgOut = cv2.GaussianBlur(gray_image, (5,5),0)
imgOut = cv2.GaussianBlur(gray_image, (13,13),0)

# Edge detection using the Sobel Operator. Sobel  Sobel operator detects horizontal and vertical edges by multiplying each pixel by the following two kernels
sobelHorizontal = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5) # y dir

sobelSum = sobelHorizontal + sobelVertical

#canny edge detection
canny = cv2.Canny(gray_image, 100, 200)

nrows = 4 # number of rows in window
ncols = 2 # number of columns in window

# Subplotting images in window
plt.subplot(nrows, ncols,1), plt.imshow(orig_img,) 
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2), plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

# Filtering images with above kernel. 5x5 and 13x13 sizes
plt.subplot(nrows, ncols,3), plt.imshow(imgOut, cmap = 'gray')
plt.title('5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4), plt.imshow(imgOut, cmap = 'gray')
plt.title('13x13'), plt.xticks([]), plt.yticks([])

# Edge Detection
plt.subplot(nrows, ncols,5), plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,6), plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,7), plt.imshow(sobelSum, cmap = 'gray')
plt.title('SobelSum'), plt.xticks([]), plt.yticks([])

# CANNY Edge Detection
plt.subplot(nrows, ncols,8), plt.imshow(canny, cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])

plt.show() #calling method
