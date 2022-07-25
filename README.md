# 1.ImageProcessing-GendersAndAges

<br>

In this Bank I work for Broadcast to detect the ages and genders of People who have died in movies.

<br>

If you dont know this Picture Please go to this link and after read that comeback.

<br>

![1A](https://user-images.githubusercontent.com/109248678/180850069-7381fdc1-49e7-47b7-ae61-db7530e35f03.jpg)

<br>

## :blush:Step One:(Familiarity with "prototxt" & "caffemodel" terms ):blush:</b>

<br>

Download these files :"age_deploy.prototxt" & "age_gender_mean.binaryproto" & "age_net.caffemodel" & "gender_net.caffemodel" & "gender_deploy.prototxt" .

<br>

### prototxt:

<br>

A PROTOTXT file is a prototype machine learning model created for use with Caffe. It contains an image classification or image segmentation model that is intended to be trained in Caffe. PROTOTXT files are used to create .CAFFEMODEL files.

<br>

### CAFFEMODEL:

<br>
A CAFFEMODEL file is a machine learning model created by Caffe. It contains an image classification or image segmentation model that has been trained using Caffe. CAFFEMODEL files are created from .PROTOTXT files.

<br>

### CAFFEMODEL:

<br>

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley. Caffe is released under the BSD 2-Clause license.

<br>

## :blush:Step two code :(Familiarity with "prototxt" & "caffemodel" terms ):blush:</b>

<br>

Download these code : ""

<br>

Code is :👇

<br>

```python
import cv2
myvideo = cv2.VideoCapture('ArmanVideo.mp4')
while True:
    ret, frame = myvideo.read() 
    cv2.imshow('myvideo', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
myvideo.release() 
cv2.destroyAllwindows()
```
