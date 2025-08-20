from ultralytics import YOLO
import cv2
import time
import math

# Load YOLO model
model = YOLO("best.pt") 
cap = cv2.VideoCapture(0)

# UI positions
start_button = (20, 20, 150, 80)
stop_button = (180, 20, 310, 80)

# Constants 
PIXELS_PER_METER = 300
DEPTH_SCALING = 5000

# State variables
tracking = False
detection_enabled = False
timer_started = False
detection_start_time = None

speed_text = "🏏 Speed: 0.00 km/h"
max_speed_text = "📈 Max: 0.00 km/h"
status_text = "Press Start"

last_position = None
last_time = None
max_speed_kmph = 0.0

def draw_ui(frame):
    cv2.rectangle(frame, start_button[:2], start_button[2:], (0, 200, 0), -1)
    cv2.putText(frame, "START", (start_button[0]+20, start_button[1]+45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

    cv2.rectangle(frame, stop_button[:2], stop_button[2:], (0, 0, 200), -1)
    cv2.putText(frame, "STOP", (stop_button[0]+20, stop_button[1]+45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    cv2.putText(frame, f"Status: {status_text}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(frame, speed_text, (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    cv2.putText(frame, max_speed_text, (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), 2)

def on_mouse(event, x, y, flags, param):
    global tracking, detection_enabled, speed_text, max_speed_text
    global max_speed_kmph, status_text, last_position, last_time
    global timer_started, detection_start_time

    if event == cv2.EVENT_LBUTTONDOWN:
        if start_button[0] <= x <= start_button[2] and start_button[1] <= y <= start_button[3]:
            print("▶️ Start pressed")
            tracking = True
            detection_enabled = True
            timer_started = False
            detection_start_time = None
            last_position = None
            last_time = None
            max_speed_kmph = 0.0
            speed_text = "🏏 Speed: 0.00 km/h"
            max_speed_text = "📈 Max: 0.00 km/h"
            status_text = "Waiting for ball..."

        elif stop_button[0] <= x <= stop_button[2] and stop_button[1] <= y <= stop_button[3]:
            print("⏹️ Stop pressed")
            tracking = False
            detection_enabled = False
            timer_started = False
            detection_start_time = None
            last_position = None
            last_time = None
            status_text = "Stopped"
            speed_text = "🏏 Speed: 0.00 km/h"

cv2.namedWindow("YOLOv8 Ball Speed Tracker")
cv2.setMouseCallback("YOLOv8 Ball Speed Tracker", on_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if detection_enabled:
        results = model(frame)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        ball_found = False

        for box in boxes:
            conf = float(box.conf[0])
            if conf > 0.1:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bbox_height = y2 - y1
                z = DEPTH_SCALING / bbox_height if bbox_height > 0 else 0

                current_position = (cx, cy, z)
                current_time = time.time()
                ball_found = True

                if not timer_started:
                    timer_started = True
                    detection_start_time = current_time
                    status_text = "Tracking..."

                if last_position and last_time:
                    dx = current_position[0] - last_position[0]
                    dy = current_position[1] - last_position[1]
                    dz = current_position[2] - last_position[2]
                    distance_pixels = math.sqrt(dx**2 + dy**2 + dz**2)
                    distance_meters = distance_pixels / PIXELS_PER_METER
                    duration = current_time - last_time

                    if duration > 0:
                        speed_mps = distance_meters / duration
                        speed_kmph = speed_mps * 3.6
                        speed_text = f"🏏 Speed: {speed_kmph:.2f} km/h"

                        if speed_kmph > max_speed_kmph:
                            max_speed_kmph = speed_kmph
                            max_speed_text = f"📈 Max: {max_speed_kmph:.2f} km/h"

                        print(speed_text)

                last_position = current_position
                last_time = current_time
                break

        if not ball_found:
            speed_text = "🏏 Speed: 0.00 km/h"

        # Stop detection 3 seconds after ball first detected
        if timer_started and (time.time() - detection_start_time) >= 3:
            print("⏳ Auto-stopped after 3 seconds")
            detection_enabled = False
            tracking = False
            timer_started = False
            detection_start_time = None
            last_position = None
            last_time = None
            status_text = "Finished - Press Start"

    else:
        annotated_frame = frame.copy()

    draw_ui(annotated_frame)
    cv2.imshow("YOLOv8 Ball Speed Tracker", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
