# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:04:15 2017

@author: me
"""


import numpy as np
import cv2
# define the `casacade for face and eyes 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade =eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
# starting point of the rectangle is (x,y) and ending point is (w,h) in the case of face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#we are making it a blue rectangle so it will be 255.0.0
        roi_gray = img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
 #now we will do the detection in the case of eyes similarly as I have done for        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    if(cv2.waitKey(1)==ord('q')):
        break;

cap.release()
cv2.destroyAllWindows()
