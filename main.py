import cv2
import mediapipe as mp
import time

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Puntos clave para detectar ojos y cabeza
LEFT_EYE = [159, 145]
RIGHT_EYE = [386, 374]
NOSE_TIP = 1

# Variables de conteo
micro_sleeps = []
head_nods = 0
last_blink_time = time.time()
prev_nose_y = None

# Lista para almacenar los tiempos de parpadeo
blink_timestamps = []
BLINK_THRESHOLD = 3
TIME_WINDOW = 10
last_micro_sleep_time = 0  # Para evitar agregar múltiples micro sueños seguidos
last_head_nod_time = 0  # Para evitar múltiples cabeceos seguidos
HEAD_NOD_THRESHOLD = 0.03  # Umbral para detectar cabeceo
MICRO_SLEEP_DISPLAY_TIME = 5  # Duración de la alerta en pantalla


def euclidean_distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_dist = euclidean_distance(face_landmarks.landmark[LEFT_EYE[0]], face_landmarks.landmark[LEFT_EYE[1]])
            right_eye_dist = euclidean_distance(face_landmarks.landmark[RIGHT_EYE[0]], face_landmarks.landmark[RIGHT_EYE[1]])
            nose_y = face_landmarks.landmark[NOSE_TIP].y

            # Detectar parpadeo
            if left_eye_dist < 0.02 and right_eye_dist < 0.02:
                if current_time - last_blink_time > 0.1:
                    blink_timestamps.append(current_time)
                    last_blink_time = current_time

            # Filtrar parpadeos en los últimos 15 segundos
            blink_timestamps = [t for t in blink_timestamps if current_time - t <= TIME_WINDOW]

            # Detectar micro sueño
            if len(blink_timestamps) >= BLINK_THRESHOLD and (current_time - last_micro_sleep_time >= TIME_WINDOW):
                micro_sleeps.append(current_time)
                last_micro_sleep_time = current_time  # Permitir nuevo micro sueño tras 15 segundos
                print("Micro sueno detectado!")

            # Detectar cabeceo
            if prev_nose_y is not None:
                if nose_y > prev_nose_y + HEAD_NOD_THRESHOLD and (current_time - last_head_nod_time > 1.5):
                    head_nods += 1
                    last_head_nod_time = current_time
                    print(f"Cabezazo detectado: {head_nods}")

            prev_nose_y = nose_y

    # Posicionar el texto en la parte inferior derecha
    height, width, _ = frame.shape
    offset_x = width - 300
    offset_y = height - 50

    cv2.putText(frame, f"Micro suenos: {len(micro_sleeps)}", (offset_x, offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cabezazos: {head_nods}", (offset_x, offset_y - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar alerta de micro sueño solo durante 5 segundos
    if len(micro_sleeps) > 0 and (current_time - micro_sleeps[-1] <= MICRO_SLEEP_DISPLAY_TIME):
        text_x = width // 3
        text_y = height // 4
        cv2.putText(frame, "Micro sueno detectado!", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Detección de Somnolencia", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
