import cv2
import mediapipe as mp


# 1. Setup MediaPipe Face Mesh

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
eyes_and_mouth =mp_drawing.DrawingSpec( color=(203,37,209),thickness=1,circle_radius=1 )
contour_style = mp_drawing.DrawingSpec(color=(185, 31,191 ), thickness=1, circle_radius=1)

# Initialize the Face Mesh model
# refine_landmarks=True gives more detail around eyes/lips
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Setup Webcam
cap = cv2.VideoCapture(0)  # '0' is usually the default webcam

print("Press 'Esc' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 3. Pre-process the image
    # MediaPipe requires RGB input, but OpenCV gives us BGR.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 4. Detect the face
    results = face_mesh.process(image)

    # Convert back to BGR to draw on it with OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 5. Draw the lines
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=contour_style
            )
            # Option B: Draw the outer contours (The "Shape" - Eyes, Lips, Face Oval)
        #     mp_drawing.draw_landmarks(
        #         image=image,
        #         landmark_list=face_landmarks,
        #         connections=mp_face_mesh.FACEMESH_LEFT_EYE,
        #         landmark_drawing_spec=None,
        #         connection_drawing_spec=eyes_and_mouth
        #     )
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=eyes_and_mouth
        # )
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_LIPS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=eyes_and_mouth
        # )
    # 6. Show the result
    cv2.imshow('Face Shape Tracker', image)

    # Exit if 'Esc' is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()