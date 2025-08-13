import os
import time
import shutil
import firebase_admin
from firebase_admin import credentials, storage
from ultralytics import YOLO

# ----------------- CONFIG -----------------
FIREBASE_BUCKET = "your-project-id.appspot.com"
FIREBASE_KEY_FILE = "firebase_key.json"
NEW_IMG_FOLDER = "new_images"
DATASET_PATH = "dataset"
BATCH_TRIGGER = 50
IMG_SIZE = 640

# ----------------- Firebase Setup -----------------
cred = credentials.Certificate(FIREBASE_KEY_FILE)
firebase_admin.initialize_app(cred, {
    'storageBucket': FIREBASE_BUCKET
})
bucket = storage.bucket()

# ----------------- YOLO Model -----------------
model = YOLO("best.pt")  # Load your existing model

# ----------------- Prepare Folders -----------------
os.makedirs(NEW_IMG_FOLDER, exist_ok=True)
os.makedirs(f"{DATASET_PATH}/images/train", exist_ok=True)
os.makedirs(f"{DATASET_PATH}/images/val", exist_ok=True)
os.makedirs(f"{DATASET_PATH}/labels/train", exist_ok=True)
os.makedirs(f"{DATASET_PATH}/labels/val", exist_ok=True)

# ----------------- Download Images from Firebase -----------------
def download_new_images():
    blobs = list(bucket.list_blobs(prefix="user_uploads/"))
    downloaded_files = []

    for blob in blobs:
        if blob.name.endswith((".jpg", ".png", ".jpeg")):
            filename = os.path.join(NEW_IMG_FOLDER, os.path.basename(blob.name))
            if not os.path.exists(filename):
                blob.download_to_filename(filename)
                downloaded_files.append(filename)

    return downloaded_files

# ----------------- Auto-label using YOLO -----------------
def auto_label(images):
    results = model.predict(images, save_txt=True, save_conf=False, project="runs/labels", name="latest_labels")
    label_dir = "runs/labels/latest_labels/labels"

    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(label_dir, base + ".txt")
        if os.path.exists(label_file):
            shutil.move(img_path, f"{DATASET_PATH}/images/train/{os.path.basename(img_path)}")
            shutil.move(label_file, f"{DATASET_PATH}/labels/train/{os.path.basename(label_file)}")

    shutil.rmtree("runs/labels", ignore_errors=True)

# ----------------- Fine-tuning -----------------
def fine_tune_yolo():
    print("Starting fine-tuning...")
    model.train(
        data="data.yaml",
        epochs=5,
        imgsz=IMG_SIZE,
        batch=8,
        resume=True
    )
    print("Fine-tuning complete.")

# ----------------- Main Loop -----------------
if __name__ == "__main__":
    while True:
        new_files = download_new_images()
        print(f"Downloaded {len(new_files)} new images.")

        if new_files:
            auto_label(new_files)

        total_new = len(os.listdir(f"{DATASET_PATH}/images/train"))
        print(f"Total labeled images ready for training: {total_new}")

        if total_new >= BATCH_TRIGGER:
            fine_tune_yolo()
            # Move some images to val set
            for i, file in enumerate(os.listdir(f"{DATASET_PATH}/images/train")[:10]):
                shutil.move(f"{DATASET_PATH}/images/train/{file}", f"{DATASET_PATH}/images/val/{file}")
                label_file = file.rsplit('.', 1)[0] + ".txt"
                if os.path.exists(f"{DATASET_PATH}/labels/train/{label_file}"):
                    shutil.move(f"{DATASET_PATH}/labels/train/{label_file}", f"{DATASET_PATH}/labels/val/{label_file}")

        time.sleep(300)  # Check every 5 min
