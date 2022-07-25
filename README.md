# 1.ImageProcessing-GendersAndAges

<br>

In this Bank I work for Broadcast to detect the ages and genders of People who have died in movies. This program say gender and age of subject.

<br>


https://user-images.githubusercontent.com/109248678/180857677-8edc2804-2f91-4c87-9295-885ef513c10a.mp4


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

Download these code : "1.Realtime Age-Gender Detection.py"

<br>

Code is :👇

<br>

```python
import cv2
import face_recognition
webcam_video_stream = cv2.VideoCapture(0)
all_face_locations = []

while True:
    ret , current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame , (0,0) , fx=0.25 , fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small , number_of_times_to_upsample=2,model='hog')
    for index , current_face_location in enumerate(all_face_locations):
        top , right , bottom , left = current_face_location
        top = top*4
        right = right*4
        bottom = bottom*4
        left = left*4
        current_face_image = current_frame[top:bottom , left:right]

        #The ‘AGE_GENDER_MODEL_MEAN_VALUES’ using numpy. mean()        
        AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
        current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
        
        gender_label_list = ['Male', 'Female']
       
        gender_protext = "gender_deploy.prototxt"
        gender_caffemodel = "gender_net.caffemodel"
      
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        
        gender_cov_net.setInput(current_face_image_blob)
        
        gender_predictions = gender_cov_net.forward()
        
        gender = gender_label_list[gender_predictions[0].argmax()]
        
        
        age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        age_protext = "age_deploy.prototxt"
        age_caffemodel = "age_net.caffemodel"
        
        age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        
        age_cov_net.setInput(current_face_image_blob)
        
        age_predictions = age_cov_net.forward()
        
        age = age_label_list[age_predictions[0].argmax()]
              
        cv2.rectangle(current_frame,(left,top),(right,bottom),(0,0,255),2)
            
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, gender+" "+age+"yrs", (left,bottom+20), font, 0.5, (0,255,0),1)
    
    cv2.imshow("Webcam Video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam_video_stream.release()
cv2.destroyAllWindows()      

```
#### EX11_Leve :
- [ ] Simple! 
- [ ] Intermediate!
- [x] Hard!

<br>

## :blush:Step Three :( use this code for images ):blush:</b>

<br>

download "2.Image Age-Gender Detection.py" file after that download these image "pic1.jpg" & "pic2.jpg" & "pic3.jpg" . 

<br>

this is good site to learn blob : [Learn about blob ](https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)
