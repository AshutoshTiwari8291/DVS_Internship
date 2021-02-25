# -*- coding: utf-8 -*-
"""
@author: Ashutosh
"""

def odd():
    import cv2
    face_cascade=cv2.CascadeClassifier('E:\\Games\\Internship\\python\\Object detection project\\openCV\\haarcascade_frontalface_default.xml')
    eye_cascade=cv2.CascadeClassifier('E:\\Games\\Internship\\python\\Object detection project\\openCV\\haarcascade_eye_tree_eyeglasses.xml')
    smileCascade=cv2.CascadeClassifier('E:\\Games\\Internship\\python\\Object detection project\\openCV\\haarcascade_smile.xml')
    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #top left cordinate,bottom right cordinate
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            font=cv2.FONT_HERSHEY_SIMPLEX
            import random
            random.random()
            cv2.putText(img,'Age:'+str(random.randint(20,25)),(30,50),font,1,(255,255,100),3,cv2.LINE_AA)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            smile = smileCascade.detectMultiScale(roi_gray,scaleFactor= 1.7,minNeighbors=22,minSize=(25, 25))
            for (x, y, w, h) in smile:
                cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 255, 0), 1)
                cv2.putText(img,'Yee..you look cute when u smile....keep smiling',(30,70),font,0.7,(0,0,255),1,cv2.LINE_AA)
                #text,start point,font,size of text,color,Thickness of text
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
def morphoanalysis():
    import cv2
    import numpy as np
    cap=cv2.VideoCapture(0)

    while True:
        _,frame=cap.read() # _ is a variable
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#hue,saturation,value
        lower_red=np.array([0,120,150])   # Change this to change HSV
        upper_red=np.array([60,255,255])
        mask=cv2.inRange(hsv,lower_red,upper_red) #it is the things which is in the range here everything comes inside or mask is now identical to the frame variable
        res=cv2.bitwise_and(frame,frame,mask=mask)# something in the fame where mask is true like 1 for white now everything is 1
        kernel=np.ones((5,5))
        erosion=cv2.erode(mask,kernel,iterations=1)
        dilation=cv2.dilate(mask,kernel,iterations=1)
        opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        closing=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        cv2.imshow('frame',frame)
        cv2.imshow('res',res)
        cv2.imshow('erosion',erosion)
        cv2.imshow('dialation',dilation)
        cv2.imshow('Opening',opening) # outside the object
        cv2.imshow('Closing',closing) # inside oject
        k=cv2.waitKey(5) & 0xFF
        if k==27:
            break
    cv2.destroyAllWindows()
    cap.release()


# Basic UI for the face detection and morphological analysis
import tkinter as t
root=t.Tk()
root.title("mygui...")
root.geometry("500x200+800+200") # To change the size (500x500) these are the x and y length of yemplate and adding '+' will give the distance of x and y axis from window
l1=t.Label(root,text="Do the object detection..",fg="pink",font=30).pack()
b1=t.Button(root,text='Object Detection',font=20,command=odd).place(x=100,y=100)
b2=t.Button(root,text='Morphological analysis',font=20,command=morphoanalysis).place(x=100,y=150)
root.mainloop()
