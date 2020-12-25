#!/usr/bin/env python
# coding: utf-8

# > Day 1: Face Detection with OpenCV and Deep Learning
# 
# - toc: true 
# - badges: true
# - comments: true
# - categories: [ Deep Learning, Computer Vision]
# - image: images/chart-preview.png

# # Day1 Tutorial: Computer Vision Course

# Required arrguments:
#     
#     --image: The path to the input image.
#     --prototxt: The path to the Caffe prototxt file.
#     --model: The path to the pretrained Caffe model
#     --confidence:
# 

# In[3]:


import numpy as np
import argparse
import cv2
import os


# In[10]:


# construct the argument parse and parse the arguments
#ap.add_argument('--name', '-n', default='foo', help='foo')
ap = argparse.ArgumentParser(description='Fooo')
ap.add_argument("-i", "--image",                #default = "/home/imagda/_coursera/pyimage/Day1/deep-learning-face-detection/face_detection01.jpg",
                default = "/home/imagda/sims-blog/_notebooks/images/face_detection05.jpg", required=True,
    help="path to input image")
ap.add_argument("-p", "--prototxt", default = "/home/imagda/_coursera/pyimage/Day1/deep-learning-face-detection/deploy.prototxt.txt",
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default = "/home/imagda/_coursera/pyimage/Day1/deep-learning-face-detection/res10_300x300_ssd_iter_140000.caffemodel",#required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.20,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args()) #fo *.py use args = vars(ap.parse_args())


# In[ ]:


# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
print(h, w)


# In[ ]:


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("...done")


# In[ ]:


# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
print("...done")


# In[ ]:


# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.imwrite("face_detection05_t00.jpg", image)
#cv2.waitKey(0)


# In[13]:


#!jupyter nbconvert --to script 2020-12-22-Day1_Face_Detection_OpenCV.ipynb


# References:
# 
# <https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/>
# 
# <https://realpython.com/command-line-interfaces-python-argparse/>
# 
# <https://medium.com/@data.scientist/ipython-trick-how-to-use-argparse-in-ipython-notebooks-a07423ab31fc>
