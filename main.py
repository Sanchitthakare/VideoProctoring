import cv2
import torch
from ultralytics import YOLO
import numpy as np
import dlib
import math
import time
import datetime
import threading
import pyaudio
from skimage.feature import local_binary_pattern


device = 'cpu'

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

model = YOLO("yolo11x.pt")
model.to(device)
class_names = model.names


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


LEFT_EYE_INDICES = list(range(36, 42))
RIGHT_EYE_INDICES = list(range(42, 48))
MOUTH_INDICES = list(range(60, 68))  


gaze_duration_threshold = 3      
face_absence_threshold = 3          
mouth_open_duration_threshold = 3  
MOUTH_OPEN_THRESHOLD = 15          
SPOOF_ENTROPY_THRESHOLD = 4.0      
NOISE_THRESHOLD = 500               

HEAD_DOWN_PERSISTENCE_THRESHOLD = 30  
EYE_CLOSED_THRESHOLD = 0.25           
EYE_CLOSED_DURATION_THRESHOLD = 3    


start_looking_away_time = None
face_absence_start = None
start_mouth_open_time = None
head_down_start = None
eyes_closed_start = None

running = True 

font = cv2.FONT_HERSHEY_SIMPLEX


model_points = np.array([
    (0.0, 0.0, 0.0),           
    (0.0, -330.0, -65.0),        
    (-225.0, 170.0, -135.0),      
    (225.0, 170.0, -135.0),      
    (-150.0, -150.0, -125.0),    
    (150.0, -150.0, -125.0)     
])


def log_event(message):
    """Print the event with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}: {message}")

def eye_on_mask(mask, indices, shape):
    points = [shape.part(i) for i in indices]
    points = np.array([[pt.x, pt.y] for pt in points], dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = min(points[1][1], points[2][1])
    r = points[3][0]
    b = max(points[4][1], points[5][1])
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    x_ratio = (end_points[0] - cx) / (cx - end_points[2] + 1e-5)
    y_ratio = (cy - end_points[1]) / (end_points[3] - cy + 1e-5)
    if x_ratio > 3:
        return 1 
    elif x_ratio < 0.33:
        return 2  
    elif y_ratio < 0.33:
        return 3  
    else:
        return 0  

def contouring(thresh, mid, img, end_points, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        return 0

def process_thresh(thresh):
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):
    global start_looking_away_time
    if left == right and left != 0:
        if start_looking_away_time is None:
            start_looking_away_time = time.time()
        else:
            elapsed_time = time.time() - start_looking_away_time
            if elapsed_time >= gaze_duration_threshold:
                cv2.putText(img, 'Looking Away Detected!', (30, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                log_event("Candidate looking away for over threshold")
    else:
        start_looking_away_time = None

def get_head_pose(shape, img_size):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     
        (shape.part(8).x, shape.part(8).y),        
        (shape.part(36).x, shape.part(36).y),       
        (shape.part(45).x, shape.part(45).y),     
        (shape.part(48).x, shape.part(48).y),      
        (shape.part(54).x, shape.part(54).y)     
    ], dtype="double")

    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
    nose_end_point2D, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    return image_points, rotation_vector, translation_vector, nose_end_point2D, camera_matrix

def calculate_mouth_open(shape):
    top_lip = shape.part(62)
    bottom_lip = shape.part(66)
    distance = math.sqrt((top_lip.x - bottom_lip.x) ** 2 + (top_lip.y - bottom_lip.y) ** 2)
    return distance


def compute_ear(eye):

    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
  
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def detect_face_spoofing(face_roi):
    """
    Compute LBP for the face region and calculate histogram entropy.
    Higher entropy suggests a potential spoof.
    """
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(face_roi, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-5)
    entropy = -np.sum(hist * np.log(hist + 1e-7))
    if entropy > SPOOF_ENTROPY_THRESHOLD:
        return "Spoof"
    else:
        return "Real"


def audio_monitor():
    """
    Continuously monitor ambient audio using PyAudio.
    If the RMS exceeds the threshold, print an alert.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                    input=True, frames_per_buffer=1024)
    while running:
        try:
            data = stream.read(1024, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms > NOISE_THRESHOLD:
                log_event("High ambient noise detected")
            time.sleep(0.5)
        except Exception as e:
            print("Audio monitor error:", e)
    stream.stop_stream()
    stream.close()
    p.terminate()

audio_thread = threading.Thread(target=audio_monitor, daemon=True)
audio_thread.start()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_size = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 
    rects = detector(gray, 0)
    if len(rects) == 0:
        if face_absence_start is None:
            face_absence_start = time.time()
        else:
            elapsed_face_absence = time.time() - face_absence_start
            if elapsed_face_absence >= face_absence_threshold:
                cv2.putText(frame, 'Face Not Detected!', (30, 60), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                log_event("Candidate face not detected for over threshold")
    else:
        face_absence_start = None

    for rect in rects:
        shape = predictor(gray, rect)

     
        x = max(rect.left(), 0)
        y = max(rect.top(), 0)
        w = rect.width()
        h = rect.height()
        face_roi = gray[y:y+h, x:x+w]
        spoof_status = detect_face_spoofing(face_roi)
        cv2.putText(frame, f'Face: {spoof_status}', (x, y - 10), font, 0.5,
                    (0, 255, 0) if spoof_status == "Real" else (0, 0, 255), 2)
        if spoof_status == "Spoof":
            log_event("Face spoofing detected")

    
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, LEFT_EYE_INDICES, shape)
        mask, end_points_right = eye_on_mask(mask, RIGHT_EYE_INDICES, shape)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)
        eyes = cv2.bitwise_and(frame, frame, mask=mask)
        mask_inv = cv2.bitwise_not(mask)
        eyes[mask_inv == 255] = (255, 255, 255)
        left_eye_coords = np.array([[shape.part(i).x, shape.part(i).y] for i in LEFT_EYE_INDICES])
        right_eye_coords = np.array([[shape.part(i).x, shape.part(i).y] for i in RIGHT_EYE_INDICES])
        mid = (left_eye_coords[3][0] + right_eye_coords[0][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        _, thresh_eye = cv2.threshold(eyes_gray, 75, 255, cv2.THRESH_BINARY)
        thresh_eye = process_thresh(thresh_eye)
        eyeball_pos_left = contouring(thresh_eye[:, 0:mid], mid, frame, end_points_left)
        eyeball_pos_right = contouring(thresh_eye[:, mid:], mid, frame, end_points_right, True)
        print_eye_pos(frame, eyeball_pos_left, eyeball_pos_right)


        image_points, rotation_vector, translation_vector, nose_end_point2D, camera_matrix = get_head_pose(shape, img_size)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        try:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            angle = int(math.degrees(math.atan(m)))
        except:
            angle = 90

       
        if angle > 30:
            cv2.putText(frame, 'Head Down', (50, 80), font, 1, (0, 0, 255), 2)
            if head_down_start is None:
                head_down_start = time.time()
            else:
                if time.time() - head_down_start >= HEAD_DOWN_PERSISTENCE_THRESHOLD:
                    cv2.putText(frame, 'Head Down for 30+ sec!', (50, 110), font, 1, (0, 0, 255), 2)
                    log_event("Head down detected for over 30 seconds")
        else:
            head_down_start = None
          
            if angle < -15:
                cv2.putText(frame, 'Head Up', (50, 80), font, 1, (0, 0, 255), 2)
                log_event("Suspicious head up detected")
            else:
                cv2.putText(frame, 'Head Straight', (50, 80), font, 1, (0, 255, 0), 2)

   
        mouth_distance = calculate_mouth_open(shape)
        if mouth_distance > MOUTH_OPEN_THRESHOLD:
            if start_mouth_open_time is None:
                start_mouth_open_time = time.time()
            else:
                if time.time() - start_mouth_open_time >= mouth_open_duration_threshold:
                    cv2.putText(frame, 'Mouth Open Detected!', (50, 140), font, 1, (0, 0, 255), 2)
                    log_event("Mouth open detected for extended duration")
        else:
            start_mouth_open_time = None

     
        left_eye = np.array([[shape.part(i).x, shape.part(i).y] for i in LEFT_EYE_INDICES])
        right_eye = np.array([[shape.part(i).x, shape.part(i).y] for i in RIGHT_EYE_INDICES])
        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        if ear < EYE_CLOSED_THRESHOLD:
            if eyes_closed_start is None:
                eyes_closed_start = time.time()
            else:
                if time.time() - eyes_closed_start >= EYE_CLOSED_DURATION_THRESHOLD:
                    cv2.putText(frame, 'Eyes Closed / Drowsiness Detected!', (50, 170), font, 1, (0, 0, 255), 2)
                    log_event("Drowsiness/eyes closed detected for over threshold")
        else:
            eyes_closed_start = None

      
        for i in range(68):
            pt = shape.part(i)
            cv2.circle(frame, (pt.x, pt.y), 1, (0, 255, 0), -1)

    results = model.predict(frame, device=device, verbose=False)
    detections = results[0].boxes.data.cpu().numpy() if results and results[0].boxes.data is not None else []
    person_count = 0
    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection[:6]
        class_id = int(class_id)
        label = class_names[class_id]
        confidence = float(score)
        if confidence > 0.5:
            if label == 'person':
                person_count += 1
                color = (0, 255, 0)
            elif label in ['earphone', 'cell phone', 'laptop', 'tv']:
                color = (255, 0, 0)
                cv2.putText(frame, 'Electronic Device Detected!', (50, 200), font, 1, (0, 0, 255), 2)
                log_event(f"Banned device detected: {label}")
            else:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), font, 0.5, color, 2)
    if person_count > 1:
        cv2.putText(frame, 'Multiple Persons Detected!', (50, 230), font, 1, (0, 0, 255), 2)
        log_event("Multiple persons detected")

 
    cv2.imshow('Advanced Real-Time Proctoring System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False 
        break


cap.release()
cv2.destroyAllWindows()
