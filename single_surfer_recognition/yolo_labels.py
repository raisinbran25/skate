import os
import pandas as pd
from ultralytics import YOLO
import torch

# --- Configuration ---
# Must match the folder where your captured images are stored
IMAGE_DIR = "single_surfer_recognition/training_photos" 
# Output file for the labels
OUTPUT_CSV = "single_surfer_recognition/teacher_labels.csv" 
# The COCO class ID for 'person' is 0 (this is the only class we care about)
TARGET_CLASS_ID = 0 
# Confidence threshold: only accept detections this confident or higher
CONF_THRESHOLD = 0.5 

def generate_teacher_labels_csv(image_dir, output_csv, target_class_id, conf_threshold):
    """
    Loads a teacher model, runs inference on images, extracts the single best 
    'person' detection, and saves the results to a CSV file.
    
    The output bounding boxes are saved in the normalized YOLO format: 
    [x_center, y_center, w, h] (0 to 1 range).
    """
    print("🌊 Starting Automated Bounding Box Labeler...")
    
    # 1. Load the Teacher Model
    try:
        # Using yolov8n for speed and decent performance
        model = YOLO('yolov8n.pt') 
        print("✅ Teacher Model (YOLOv8n) loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading YOLO model. Ensure 'ultralytics' is installed and has internet access: {e}")
        return

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))])
    total_images = len(image_files)

    if total_images == 0:
        print(f"⚠️ No images found in {image_dir}. Please run the frame capture script first.")
        return

    # List to store all rows for the final CSV
    label_records = []
    
    print(f"Processing {total_images} images...")

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        
        # 2. Run Inference
        # Running in CPU mode is often necessary if you're multitasking; remove 'cpu' if you have a GPU
        results = model(image_path, conf=conf_threshold, classes=[target_class_id], verbose=False) 
        
        # Default label if no person is detected
        label_data = [image_file, 0.0, 0.0, 0.0, 0.0, 0.0] 
        
        # 3. Extract the Single Best 'Person' Detection
        
        # Iterate over results (should only be one batch/image)
        for r in results:
            # Filter results to include only the highest confidence 'person' box
            best_conf = 0.0
            best_box_normalized = None
            
            # The 'r.boxes' object contains all detections
            if r.boxes:
                for box in r.boxes:
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    
                    # Ensure it's the target class ('person') and has higher confidence
                    if cls == target_class_id and conf > best_conf:
                        best_conf = conf
                        
                        # Get normalized coordinates [x_center, y_center, w, h]
                        # xywhn converts to normalized center-width-height format (0-1 range)
                        # We use .view(-1) to flatten the 1x4 tensor to a list of 4 values
                        box_tensor = box.xywhn.view(-1).cpu().numpy()
                        best_box_normalized = box_tensor.tolist()
                        
            # If a detection was found, update the label data
            if best_box_normalized:
                x_c, y_c, w, h = best_box_normalized
                
                # Format: [filename, x_c, y_c, w, h, confidence]
                label_data = [image_file, x_c, y_c, w, h, best_conf]
                
                # Since we only want the single best person/surfer, we break after finding it
                break

        label_records.append(label_data)
        
        if (i + 1) % 50 == 0:
            print(f"  -> Processed {i + 1}/{total_images} images. Latest confidence: {label_data[5]:.3f}")

    # 4. Save to CSV
    df = pd.DataFrame(label_records, columns=['filename', 'x_center', 'y_center', 'width', 'height', 'confidence'])
    df.to_csv(output_csv, index=False)
    
    print(f"\n✨ Label generation complete!")
    print(f"Total labels saved: {len(label_records)}")
    print(f"Output CSV path: {os.path.abspath(output_csv)}")

# --- Run the Script ---
if __name__ == "__main__":
    # Create the base directory if it doesn't exist (e.g., single_surfer_recognition)
    os.makedirs(os.path.dirname(IMAGE_DIR), exist_ok=True)
    
    generate_teacher_labels_csv(IMAGE_DIR, OUTPUT_CSV, TARGET_CLASS_ID, CONF_THRESHOLD)