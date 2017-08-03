# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:04:15 2017

@author: me
"""
#Doing the face detection using opencv and haar cascade
# Import the pakages 
import numpy as np
import cv2
# Now define the `casacade for face 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Setting up the webcam
cap = cv2.VideoCapture(0)
# Now making the code to detect the faces 
while 1:
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,a,w,b) in faces:
        cv2.rectangle(image,(x,a),(x+w,a+b),(255,0,0),2)
#I am  making it a blue rectangle to detect the face so it is 255.0.0
        roi_gray = image[a:a+b, x:x+w]
        roi_color = image[a:a+b, x:x+w]
    cv2.imshow('img',image)
    if(cv2.waitKey(1)==ord('q')):
        break;

cap.release()
cv2.destroyAllWindows()
