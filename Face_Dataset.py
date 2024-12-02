import cv2 #importing openCV

camera=cv2.VideoCapture(0) #to capture video through camera using openCV
#set() gives width and height in terms of pixels
camera.set(3,640) #for width(3)
camera.set(4, 480) #for height(4)

#has complex classifiers like AdaBoost which allows negative input(non-face) to be quickly discarded while spending more computation on promising or positive face-like regions.
face=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') #to detect the face

face_id=input("\n\tEnter user ID and press enter:") #in order to get recognised later
print("\tCapturing face....")

i=0 #keeps a count of the number of images

while(True): #to get images for dataset
    ret, img=camera.read() #to capture images using the camera, returns boolean and data
    
    #grayscale compresses an image to its barest minimum pixel, thereby making it easy for visualization
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting image to gray color
    
    #if a rectangle is found, it returns Rect(x, y, w, h)
    faces=face.detectMultiScale(gray, 1.3, 5) #multiscale detection of gray image with dimensions    

    for(a, b, c, d) in faces: #to store images in dataset
        #to draw the rectangle in the original image that we found out in the frame with 
        #parameters as the image, start of (x, y) then the width and height as 
        #(x+w, y+h) and finally the color in BGR
        cv2.rectangle(img, (a, b), (a+c, b+d), (255, 0, 0))
        i+=1
        
        #writing into the dataset, first the name of the images in dataset and then the image
        cv2.imwrite("./dataset/User."+ str(face_id) +"."+ str(i) +".jpg", gray[b:b+d, a:a+c])
        
        #to display the image that is scanned 
        cv2.imshow("image", img)

    #takes in miliseconds after which it will close, if argument is 0, then it will run until a key is pressed
    x=cv2.waitKey(20) & 0xff
    #if x==97:
        #break
    if i>=150:
        break

print("\n\tExiting Program")
camera.release()
cv2.destroyAllWindows()