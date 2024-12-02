import cv2

recognizer=cv2.face.LBPHFaceRecognizer.create()
recognizer.read('./Trainer/trainer.yml')
cascadePath="./haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascadePath)

font=cv2.FONT_HERSHEY_TRIPLEX

#initiate id counter
id=0

names=[j for j in range(181)] 
#names["Chinmay"]

#initialize and start realtime video capture
cam=cv2.VideoCapture(0)
cam.set(3, 640) #set video width
cam.set(4, 480) #set video height

#define min window size to be recognized as a face of minimum width and height
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

i=0
while True:

    ret, img=cam.read()
   
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting image to gray color

    faces=faceCascade.detectMultiScale(  #detect faces using haar classifier and storing it
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)), #minimum width and minimum height
        )

    for(x,y,w,h) in faces: #making 4 different edges

        #image, top-left, bottom-right, BGR, thickness
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 
        i+=1
        
        #Region Of Interest - height, width
        id, confidence=recognizer.predict(gray[y:y+h,x:x+w]) 

        #check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 65): #if the picture is recognised
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else: #if the picture is not recognised
            id = "unknown" 
            confidence = "  {0}%".format(round(100 - confidence))
        
        #image, string, positioning, font, font scale factor(set to default), thickness
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera', img) #showing the camera

    k=cv2.waitKey(16) & 0xff #To extract the ASCII value of the pressed key
    if k==97: #ASCII value of 'a' from the keyboard
        break 
    elif i>=150:
        break 
    
#do a bit of cleanup
print("\n\tExiting Program")
cam.release()
cv2.destroyAllWindows()
