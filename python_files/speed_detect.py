from ultralytics import YOLO
import cv2
import time
import math

# Load YOLO model
model = YOLO("best.pt") 
cap = cv2.VideoCapture(0)

# Constants
PIXELS_PER_METER = 300
DEPTH_SCALING = 5000

# Speed state
last_position = None
last_time = None
max_speed_kmph = 0.0

# UI text
speed_text = "🏏 Speed: 0.00 km/h"
max_speed_text = "📈 Max: 0.00 km/h"
status_text = "Detecting..."

def draw_ui(frame):
    cv2.putText(frame, f"Status: {status_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, speed_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    cv2.putText(frame, max_speed_text, (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    draw_ui(annotated_frame)
    cv2.imshow("YOLOv8 Ball Speed Tracker", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
