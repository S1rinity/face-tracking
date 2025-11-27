import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Перевірте індекс камери (0 або 1)
camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Не вдалося відкрити камеру")
    exit()

# Шрифт тексту
font = cv2.FONT_ITALIC

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)
        text = f"X:{x} Y:{y}"
        text_y = y - 10 if y - 10 > 10 else y + h + 20
        cv2.putText(frame, text, (x, text_y), font, 1, (0, 0, 255), 3)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()