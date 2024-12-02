import cv2 #importing openCV
import numpy as np #to perform mathematical operations
from PIL import Image #to get the segment Image
import os 

#path for face image database
path='./Dataset'

recognizer=cv2.face.LBPHFaceRecognizer.create()

detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#function to get the images and label data
def getImagesAndLabels(path):
    imagePaths=[os.path.join(path, f) for f in os.listdir(path)] #to specify image path using os 
    faceSamples=[]
    ids=[]

    for imagePath in imagePaths: #for every imagePath in imagePaths
        PIL_img=Image.open(imagePath).convert('L') #convert it to grayscale L-Luminiscence
        img_numpy=np.array(PIL_img, 'uint8') #converts grayscale PIL image into numpy array
        #uint8 means unsigned integer of 8-bits and stores it in numpy array 

        #split path and file name, and further split and take the 2nd element of it
        id=int(os.path.split(imagePath)[-1].split(".")[1]) #naming conventions
        faces=detector.detectMultiScale(img_numpy) 

        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w]) #selects only ROI
            ids.append(id)

    return faceSamples, ids

print ("\n\tTraining faces. It will take a few seconds. Please wait ...")
faces, ids = getImagesAndLabels(path) #respective images and user ID
recognizer.train(faces, np.array(ids)) #trains model with corresponding faces and numpy array of ids

#save the model into trainer/trainer.yml
recognizer.write('./trainer/trainer.yml') 

#print the numer of faces trained and end program
print("\n\t{0} faces trained.".format(len(np.unique(ids))))