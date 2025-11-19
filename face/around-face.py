import cv2
from opt_einsum.backends.tensorflow import build_expression_eager

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Could not open camera")
    exit()
while True:
    ret, frame = camera.read(0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Image',frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break