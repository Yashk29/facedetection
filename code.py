import cv2
face_c=cv2.CascadeClassifier("C:/Users/YASH/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

video_c=cv2.VideoCapture(0)
while True:
    ret,video_d= video_c.read()
    col=cv2.cvtColor(video_d,cv2.COLOR_BGR2GRAY)
    faces=face_c.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(video_d,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live",video_d)
    if cv2.waitKey(12)== ord("a"):
        break
video_c.release()
