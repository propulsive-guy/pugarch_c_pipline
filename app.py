import streamlit as st
import os
import uuid
from datetime import datetime
from PIL import Image
from google.cloud import storage

# Your class mapping
class_mapping = {
    0: 'fine dust',
    1: 'garbagebag',
    2: 'liquid',
    3: 'paper_waste',
    4: 'plastic_bottles',
    5: 'plasticbags',
    6: 'stains'
}

# GCP settings (use env var GOOGLE_APPLICATION_CREDENTIALS for auth)
BUCKET_NAME = "retraining-pugarchv2"

# GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def save_image_to_gcs(image, label):
    """Save uploaded image to Google Cloud Storage."""
    # Create unique filename
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{label}/{label}_{timestamp}_{unique_id}.jpg"
    
    # Save temporarily
    temp_path = f"/tmp/{filename.split('/')[-1]}"
    image.save(temp_path)

    # Upload to GCS
    blob = bucket.blob(filename)
    blob.upload_from_filename(temp_path)
    
    return f"gs://{BUCKET_NAME}/{filename}"

# Streamlit UI
st.set_page_config(page_title="Garbage Detection Training Data Collector", layout="centered")

st.title("♻️ Garbage Classification Data Collector")
st.write("Upload an image and specify its class to improve the YOLOv8 model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Dropdown for class selection
    selected_class = st.selectbox("Select the class for this image:", list(class_mapping.values()))
    
    if st.button("Submit"):
        try:
            gcs_path = save_image_to_gcs(img, selected_class)
            st.success(f"✅ Image saved to {gcs_path}")
        except Exception as e:
            st.error(f"❌ Error saving image: {e}")
