import cv2
import math
import time
import threading
import csv
import os
import datetime
# import winsound # winsound is Windows-specific, commenting out for Colab

import tkinter as tk
from tkinter import Label, Button, Frame

from PIL import Image, ImageTk
from ultralytics import YOLO

import mediapipe as mp

model = YOLO(
    r"C:\Users\arya4\OneDrive\Documents\New folder\smoker-detection\runs\detect\train-6\weights\best.pt")

print("Loaded Classes:", model.names)

# ============================================================
# MEDIAPIPE HANDS AND FACE MESH
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True # Enable for more accurate face mesh landmarks
)

# ============================================================
# GLOBAL VARIABLES
# ============================================================

camera_running = False
cap = None

alert_count = 0

frame_counter = 0
frame_skip = 2

last_alert_time = 0

video_writer = None
recording = False

# ============================================================
# SMOOTHING VARIABLES
# ============================================================

prev_boxes = {}

SMOOTHING = 0.9 # Increased smoothing to reduce wiggling effect

# ============================================================
# CREATE ALERT FOLDER
# ============================================================

os.makedirs("alerts", exist_ok=True)

csv_file = open("alerts/logs.csv", "a", newline="")

csv_writer = csv.writer(csv_file)

if os.stat("alerts/logs.csv").st_size == 0:

    csv_writer.writerow([
        "Time",
        "Detected Objects",
        "Type"
    ])

# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("AI Smart Smoking Detection System")

root.geometry("1500x900")

root.configure(bg="#08111f")

# ============================================================
# HEADER
# ============================================================

header = Frame(root, bg="#111827", height=80)

header.pack(fill="x")

title = Label(
    header,
    text="🚭 AI SMART SMOKING DETECTION SYSTEM",
    bg="#111827",
    fg="#38bdf8",
    font=("Arial", 28, "bold")
)

title.pack(pady=18)

# ============================================================
# MAIN FRAME
# ============================================================

main_frame = Frame(root, bg="#08111f")

main_frame.pack(fill="both", expand=True)

# ============================================================
# VIDEO FRAME
# ============================================================

video_frame = Frame(
    main_frame,
    bg="#111827",
    bd=3,
    relief="ridge"
)

video_frame.pack(side="left", padx=20, pady=20)

video_title = Label(
    video_frame,
    text="📷 LIVE CAMERA FEED",
    bg="#111827",
    fg="white",
    font=("Arial", 18, "bold")
)

video_title.pack(pady=10)

video_label = Label(video_frame, bg="black")

video_label.pack(padx=10, pady=10)

# ============================================================
# RIGHT DASHBOARD PANEL
# ============================================================

right_panel = Frame(
    main_frame,
    bg="#111827",
    width=350
)

right_panel.pack(side="right", fill="y", padx=20, pady=20)

# ============================================================
# STATUS
# ============================================================

status_title = Label(
    right_panel,
    text="SYSTEM STATUS",
    bg="#111827",
    fg="#38bdf8",
    font=("Arial", 18, "bold")
)

status_title.pack(pady=(20, 5))

status_box = Label(
    right_panel,
    text="SAFE",
    bg="#16a34a",
    fg="white",
    width=18,
    height=2,
    font=("Arial", 28, "bold")
)

status_box.pack(pady=10)

# ============================================================
# HAND STATUS
# ============================================================

hand_label = Label(
    right_panel,
    text="✋ Hand Status: Normal",
    bg="#1e293b",
    fg="#facc15",
    font=("Arial", 16),
    width=28,
    pady=10
)

hand_label.pack(pady=10)

# ============================================================
# OBJECT STATUS
# ============================================================

object_label = Label(
    right_panel,
    text="📦 Objects: None",
    bg="#1e293b",
    fg="#38bdf8",
    font=("Arial", 16),
    width=28,
    pady=10
)

object_label.pack(pady=10)

# ============================================================
# ALERT COUNTER
# ============================================================

counter_label = Label(
    right_panel,
    text="⚠ Alerts: 0",
    bg="#1e293b",
    fg="#ef4444",
    font=("Arial", 18, "bold"),
    width=28,
    pady=15
)

counter_label.pack(pady=10)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def play_alarm():

    try:
        # winsound.Beep(1000, 300) # winsound is Windows-specific
        pass # No audible alarm on non-Windows systems

    except:
        pass


def start_recording(frame):

    global video_writer
    global recording

    if not recording:

        filename = f"alerts/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        h, w, _ = frame.shape

        video_writer = cv2.VideoWriter(
            filename,
            fourcc,
            20.0,
            (w, h)
        )

        recording = True

    video_writer.write(frame)

# ============================================================
# MAIN DETECTION
# ============================================================

def start_detection():

    global camera_running
    global cap
    global alert_count
    global frame_counter
    global last_alert_time

    if camera_running:
        return

    camera_running = True

    cap = cv2.VideoCapture(0)

    while camera_running:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frame_counter += 1

        detected_objects = []

        person_detected = False
        smoking_object_detected = False

        smoking_boxes = []

        # ====================================================
        # YOLO DETECTION
        # ====================================================

        if frame_counter % frame_skip == 0:

            results = model(
                frame,
                imgsz=960,
                conf=0.60,
                iou=0.4
            )

            for r in results:

                for box in r.boxes:

                    conf = float(box.conf[0])

                    # ====================================================
                    # HIGH CONFIDENCE
                    # ====================================================

                    if conf < 0.60:
                        continue

                    cls = int(box.cls[0])

                    label = model.names[cls].lower()

                    detected_objects.append(label)

                    # ====================================================
                    # COLORS
                    # ====================================================

                    if label == "person":

                        color = (255, 255, 0)

                        person_detected = True

                    elif label == "cigarette":

                        color = (255, 0, 255)

                        smoking_object_detected = True

                    elif label == "vape":

                        color = (0, 0, 255)

                        smoking_object_detected = True

                    else:

                        color = (0, 255, 0)

                    # ====================================================
                    # BOX COORDINATES
                    # ====================================================

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # ====================================================
                    # BOX SMOOTHING
                    # ====================================================

                    key = label

                    if key in prev_boxes:

                        px1, py1, px2, py2 = prev_boxes[key]

                        x1 = int(px1 * SMOOTHING + x1 * (1 - SMOOTHING))
                        y1 = int(py1 * SMOOTHING + y1 * (1 - SMOOTHING))
                        x2 = int(px2 * SMOOTHING + x2 * (1 - SMOOTHING))
                        y2 = int(py2 * SMOOTHING + y2 * (1 - SMOOTHING))

                    prev_boxes[key] = (x1, y1, x2, y2)

                    # ====================================================
                    # SAVE SMOKING OBJECT BOX
                    # ====================================================

                    if label in ["cigarette", "vape"]:

                        smoking_boxes.append(
                            (x1, y1, x2, y2)
                        )

                    # ====================================================
                    # DRAW BOX
                    # ====================================================

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        3
                    )

                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

        # ====================================================
        # MEDIAPIPE HAND DETECTION
        # ====================================================

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(rgb)
        face_results = face_mesh.process(rgb)

        hand_near_object = False
        object_near_mouth = False

        # ====================================================
        # HANDS DETECTED
        # ====================================================

        if hand_results.multi_hand_landmarks:

            h, w, _ = frame.shape

            for hand_landmarks in hand_results.multi_hand_landmarks:

                # ====================================================
                # INDEX FINGER TIP
                # ====================================================

                fingertip = hand_landmarks.landmark[8]

                fx = int(fingertip.x * w)
                fy = int(fingertip.y * h)

                # ====================================================
                # CLEAN HAND POINT
                # ====================================================

                cv2.circle(
                    frame,
                    (fx, fy),
                    8,
                    (0, 255, 0),
                    -1
                )

                # ====================================================
                # CHECK DISTANCE TO CIGARETTE/VAPE
                # ====================================================

                for (x1, y1, x2, y2) in smoking_boxes:

                    obj_cx = int((x1 + x2) / 2)
                    obj_cy = int((y1 + y2) / 2)

                    dist_hand_obj = math.sqrt(
                        (fx - obj_cx) ** 2 +
                        (fy - obj_cy) ** 2
                    )

                    if dist_hand_obj < 80: # Threshold for hand holding object
                        hand_near_object = True

                        cv2.putText(
                            frame,
                            "HAND HOLDING OBJECT",
                            (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )

                        root.after(
                            0,
                            lambda: hand_label.config(
                                text="✋ Hand Holding Vape/Cigarette",
                                fg="red"
                            )
                        )

                        # ====================================================
                        # CHECK IF OBJECT IS NEAR MOUTH
                        # ====================================================
                        if face_results.multi_face_landmarks:
                            for face_landmarks in face_results.multi_face_landmarks:
                                # Mouth landmarks (example indices, adjust as needed for precise mouth area)
                                # You can find full landmark list in Mediapipe documentation
                                mouth_landmarks_indices = [61, 291, 0, 17, 78, 308] # These are example mouth landmarks. More could be added.
                                mouth_coords = []
                                for idx in mouth_landmarks_indices:
                                    lm = face_landmarks.landmark[idx]
                                    mouth_coords.append((int(lm.x * w), int(lm.y * h)))

                                if mouth_coords:
                                    # Calculate the center of the mouth region
                                    mouth_cx = sum([p[0] for p in mouth_coords]) // len(mouth_coords)
                                    mouth_cy = sum([p[1] for p in mouth_coords]) // len(mouth_coords)

                                    cv2.circle(frame, (mouth_cx, mouth_cy), 5, (255, 0, 0), -1) # Visualize mouth center

                                    dist_obj_mouth = math.sqrt(
                                        (obj_cx - mouth_cx) ** 2 +
                                        (obj_cy - mouth_cy) ** 2
                                    )

                                    if dist_obj_mouth < 100: # Threshold for object near mouth
                                        object_near_mouth = True
                                        cv2.putText(
                                            frame,
                                            "OBJECT NEAR MOUTH",
                                            (30, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 255, 255),
                                            3
                                        )
                                        break # Break after finding one mouth proximity


        else:

            root.after(
                0,
                lambda: hand_label.config(
                    text="✋ Hand Status: Normal",
                    fg="#facc15"
                )
            )

        # ====================================================
        # FINAL ALERT LOGIC
        # ====================================================

        if (
            person_detected
            and smoking_object_detected
            and hand_near_object
            and object_near_mouth # New condition for accuracy
        ):

            root.after(
                0,
                lambda: status_box.config(
                    text="ALERT",
                    bg="#dc2626"
                )
            )

            # ====================================================
            # ALERT TIMER
            # ====================================================

            if time.time() - last_alert_time > 2:

                alert_count += 1

                last_alert_time = time.time()

                csv_writer.writerow([
                    datetime.datetime.now(),
                    ",".join(detected_objects),
                    "SMOKING ALERT"
                ])

                csv_file.flush()

                play_alarm()

                start_recording(frame)

                root.after(
                    0,
                    lambda: counter_label.config(
                        text=f"⚠ Alerts: {alert_count}"
                    )
                )

        else:

            root.after(
                0,
                lambda: status_box.config(
                    text="SAFE",
                    bg="#16a34a"
                )
            )

        # ====================================================
        # OBJECT LABEL UPDATE
        # ====================================================

        if detected_objects:

            object_text = ", ".join(
                list(set(detected_objects))
            )

        else:

            object_text = "None"

        root.after(
            0,
            lambda: object_label.config(
                text=f"📦 Objects: {object_text}"
            )
        )

        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)

        img = img.resize((950, 720))

        imgtk = ImageTk.PhotoImage(img)

        def update_frame():

            video_label.imgtk = imgtk

            video_label.config(
                image=imgtk
            )

        root.after(0, update_frame)

    # ====================================================
    # RELEASE CAMERA
    # ====================================================

    if cap:
        cap.release()

    if video_writer:
        video_writer.release()

# ============================================================
# BUTTONS
# ============================================================

button_frame = Frame(
    right_panel,
    bg="#111827"
)

button_frame.pack(pady=30)

# ============================================================
# START BUTTON
# ============================================================

start_btn = Button(
    button_frame,
    text="▶ START",
    command=lambda: threading.Thread(
        target=start_detection,
        daemon=True
    ).start(),
    bg="#22c55e",
    fg="white",
    font=("Arial", 18, "bold"),
    width=15,
    height=2,
    bd=0,
    cursor="hand2"
)

start_btn.grid(
    row=0,
    column=0,
    padx=20,
    pady=15
)

# ============================================================
# STOP FUNCTION
# ============================================================

def stop():

    global camera_running

    camera_running = False

    try:

        if cap:
            cap.release()

    except:
        pass

# ============================================================
# STOP BUTTON
# ============================================================

stop_btn = Button(
    button_frame,
    text="⏹ STOP",
    command=stop,
    bg="#ef4444",
    fg="white",
    font=("Arial", 18, "bold"),
    width=15,
    height=2,
    bd=0,
    cursor="hand2"
)

stop_btn.grid(
    row=1,
    column=0,
    padx=20,
    pady=15
)

# ============================================================
# EXIT FUNCTION
# ============================================================

def exit_app():

    global camera_running

    camera_running = False

    try:

        if cap:
            cap.release()

    except:
        pass

    root.destroy()

# ============================================================
# EXIT BUTTON
# ============================================================

exit_btn = Button(
    button_frame,
    text="❌ EXIT",
    command=exit_app,
    bg="#3b82f6",
    fg="white",
    font=("Arial", 18, "bold"),
    width=15,
    height=2,
    bd=0,
    cursor="hand2"
)

exit_btn.grid(
    row=2,
    column=0,
    padx=20,
    pady=15
)

# ============================================================
# FOOTER
# ============================================================

footer = Label(
    root,
    text="AI Powered Real-Time Smoking & Vape Detection System",
    bg="#111827",
    fg="#94a3b8",
    font=("Arial", 12)
)

footer.pack(fill="x", pady=5)

# ============================================================
# RUN APPLICATION
# ============================================================

root.mainloop()