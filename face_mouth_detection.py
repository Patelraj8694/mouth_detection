import cv2  


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
  

#eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")  
mouth_cascade = cv2.CascadeClassifier("haarcascade_mouth_tree.xml")

# capture frames from a camera 
cap = cv2.VideoCapture(0) 
  
# loop runs if capturing has been initialized. 
while 1:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
    #Detects the mmouth  
        mouths = mouth_cascade.detectMultiScale(roi_gray,1.8,20)
 
        for (sx, sy, sw, sh) in mouths:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
     
  
    # Display an image in a window 
    cv2.imshow('img',img) 
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(5)
    if k == 27: 
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  