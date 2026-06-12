import os
import shutil
import random

BASE = r"C:\Users\arya4\OneDrive\Documents\New folder\smoker-detection"

SOURCE = os.path.join(BASE, "validation", "Validation")

TRAIN = os.path.join(BASE, "train", "images")
VAL = os.path.join(BASE, "valid", "images")
TEST = os.path.join(BASE, "test", "images")

os.makedirs(TRAIN, exist_ok=True)
os.makedirs(VAL, exist_ok=True)
os.makedirs(TEST, exist_ok=True)

# Get all images
images = [f for f in os.listdir(SOURCE) if f.endswith((".jpg", ".png"))]

random.shuffle(images)

n = len(images)
train_split = int(n * 0.7)
val_split = int(n * 0.9)

train_imgs = images[:train_split]
val_imgs = images[train_split:val_split]
test_imgs = images[val_split:]

def move(files, dest):
    for f in files:
        src = os.path.join(SOURCE, f)
        dst = os.path.join(dest, f)
        shutil.copy(src, dst)

move(train_imgs, TRAIN)
move(val_imgs, VAL)
move(test_imgs, TEST)

# Create YOLO YAML
yaml = f"""path: .

train: train/images
val: valid/images
test: test/images

names:
  0: person
  1: cigarette
  2: pen
  3: lollipop
  4: vape
  5: stylus
  6: inhaler
"""

with open(os.path.join(BASE, "data.yaml"), "w") as f:
    f.write(yaml)

print("✅ Dataset fixed successfully!")
print("🚀 Now run:")
print("yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640")