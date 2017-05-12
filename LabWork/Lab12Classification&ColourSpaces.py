import cv2
import numpy as np
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
lefteye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')



img = cv2.imread('mcgregor.jpg',)
img2 = cv2.imread('rhcp.jpg',)

b,g,r = cv2.split(img)

orig_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orig_img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(orig_img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = orig_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		
lefteye = lefteye_cascade.detectMultiScale(gray2, 1.3, 5)
for (x,y,w,h) in lefteye:
    cv2.rectangle(orig_img2,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray2 = gray2[y:y+h, x:x+w]
    roi_color2 = orig_img2[y:y+h, x:x+w]
#cv2.imshow('img',img)

#for i in range(len(img)):
#	for j in range(len(img[i])):
#		for k in range(len(img[i][j]))

nrows = 3 # number of rows in window
ncols = 1 # number of columns in window

plt.subplot(nrows, ncols,1), plt.imshow(orig_img,) 
plt.title('Face Detection'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,2), plt.imshow(orig_img2,) 
plt.title('rhcp'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,3), plt.imshow(img, ) 
plt.title('rhcp'), plt.xticks([]), plt.yticks([])


plt.show() #calling method