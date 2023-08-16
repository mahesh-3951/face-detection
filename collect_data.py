import cv2
import urllib
import numpy as np

classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

url = "http://192.168.109.225:8080/shot.jpg"


data = []

while len(data) < 100:
    image_from_url = urllib.request.urlopen(url)   #for accessing camera
    frame = np.array(bytearray(image_from_url.read()),np.uint8)  #To store in byte form
    frame = cv2.imdecode(frame,-1)  #Decoding the image which stored in byte format
    
    face_points = classifier.detectMultiScale(frame,1.3,5)  # multiscale--> detect multiple faces
    
    if len(face_points)>0:
        for x,y,w,h in face_points:
            face_frame = frame[y:y+h+1,x:x+w+1]
            cv2.imshow("Only face",face_frame)
            if len(data)<100:                 #collect upto 100 img and append in each step
                data.append(face_frame)
                break
    cv2.putText(frame, str(len(data)),(100,100),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255)) #for naming the image
    cv2.imshow("frame",frame)
    if cv2.waitKey(30) == ord("q"):   # to stop perss q
        break
cv2.destroyAllWindows()
        
if len(data) >= 100:
    name = input("Enter Face holder name : ")
    for i in range(100):
        cv2.imwrite("images/"+name+"_"+str(i)+".jpg",data[i])
    print("Done")
else:
    print("need more data")
        
    

