import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def draw_box(frame, points, label, state, position="left"):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    padding = 20
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    color = (0,255,0) if state == "Open" else (0,0,255)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    if position == "left":
        text_x = max(10, x_min - 140) 
        text_y = y_min

    elif position == "right":
        text_x = min(frame.shape[1] - 200, x_max + 10)
        text_y = y_min

    else:
        text_x = x_min
        text_y = y_max + 30

    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


LEFT_THRESHOLD = 0.23
RIGHT_THRESHOLD = 0.23
MOUTH_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]
            mouth = [13, 14, 78, 308]

            def get_coords(index):
                lm = face_landmarks.landmark[index]
                return int(lm.x * w), int(lm.y * h)

            p1, p2, p3, p4, p5, p6 = [get_coords(i) for i in left_eye]
            left_EAR = (calculate_distance(p2, p6) + calculate_distance(p3, p5)) / (2 * calculate_distance(p1, p4))

            p1, p2, p3, p4, p5, p6 = [get_coords(i) for i in right_eye]
            right_EAR = (calculate_distance(p2, p6) + calculate_distance(p3, p5)) / (2 * calculate_distance(p1, p4))

            m1, m2, m3, m4 = [get_coords(i) for i in mouth]
            MAR = calculate_distance(m1, m2) / calculate_distance(m3, m4)

            left_eye_state = "Closed" if left_EAR < LEFT_THRESHOLD else "Open"
            right_eye_state = "Closed" if right_EAR < RIGHT_THRESHOLD else "Open"

            if abs(left_EAR - right_EAR) > 0.07:
                if left_EAR < right_EAR:
                    left_eye_state = "Closed"
                    right_eye_state = "Open"
                else:
                    right_eye_state = "Closed"
                    left_eye_state = "Open"

            mouth_state = "Open" if MAR > MOUTH_THRESHOLD else "Closed"

            left_eye_points = [get_coords(i) for i in left_eye]
            right_eye_points = [get_coords(i) for i in right_eye]
            mouth_points = [get_coords(i) for i in mouth]

            draw_box(frame, left_eye_points,
                     f"Left Eye: {left_eye_state}",
                     left_eye_state,
                     position="left")

            draw_box(frame, right_eye_points,
                     f"Right Eye: {right_eye_state}",
                     right_eye_state,
                     position="right")

            draw_box(frame, mouth_points,
                     f"Mouth: {mouth_state}",
                     mouth_state,
                     position="center")

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
