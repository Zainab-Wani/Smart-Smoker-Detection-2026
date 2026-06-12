from ultralytics import YOLO
import os
import cv2

# =========================
# PATHS
# =========================

BASE = r"C:\Users\arya4\OneDrive\Documents\New folder\smoker-detection"

IMG_DIRS = [
    os.path.join(BASE, "train/images"),
    os.path.join(BASE, "valid/images"),
    os.path.join(BASE, "test/images")
]

LABEL_DIRS = [
    os.path.join(BASE, "train/labels"),
    os.path.join(BASE, "valid/labels"),
    os.path.join(BASE, "test/labels")
]

for d in LABEL_DIRS:
    os.makedirs(d, exist_ok=True)

# =========================
# LOAD PRETRAINED MODEL
# =========================

model = YOLO("yolov8n.pt")

# COCO class mapping
coco_classes = model.names

# =========================
# AUTO LABEL FUNCTION
# =========================

def label_images(img_dir, label_dir):

    images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]

    for img_name in images:

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        results = model(img)[0]

        h, w, _ = img.shape

        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        with open(label_path, "w") as f:

            for box in results.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # filter weak detections
                if conf < 0.4:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]

                # YOLO format
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"{cls} {x_center} {y_center} {bw} {bh}\n")

        print(f"Labeled: {img_name}")

# =========================
# RUN
# =========================

for img_dir, label_dir in zip(IMG_DIRS, LABEL_DIRS):
    print(f"Processing {img_dir}")
    label_images(img_dir, label_dir)

print("DONE: Auto-labeling complete")