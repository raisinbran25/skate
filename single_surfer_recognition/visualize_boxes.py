import pandas as pd
import cv2
import os
import sys
import numpy as np

# --- 1. CONFIGURATION ---
IMAGE_DIR = "single_surfer_recognition/testing_photos"
WINDOW_NAME = "Multi-Model Prediction QA Tool"

# Define the paths for the three CSV sources
LABEL_CSVS = {
    "teacher": "single_surfer_recognition/teacher_test_labels.csv", # Green (Source of truth)
    "student": "single_surfer_recognition/student_labels.csv",  # Yellow (Trained model)
    "random": "single_surfer_recognition/random_labels.csv"    # Red (Untrained model)
}

# Define the Bounding Box Colors and VISIBILITY (BGR format for OpenCV)
# The third element in the tuple (0 or 1) is the visibility flag.
BOX_SETTINGS = {
    "teacher": ((0, 255, 0), True),  # Green (Visible)
    "student": ((0, 255, 255), True),# Yellow (Visible)
    "random": ((0, 0, 255), False)    # Red (Visible)
}

# Keys for navigation
NEXT_KEY = ord('w')
PREV_KEY = ord('q')
ESC_KEY = 27 # ESC key to quit

# Array of filenames to consider (Must match filenames in IMAGE_DIR)
SELECTED_FILES = [] 
for i in range(len(os.listdir(f"{IMAGE_DIR}"))):
    SELECTED_FILES.append(f"{i:05d}.jpg")


# --- 2. DATA LOADING AND UTILITIES ---

def load_all_labels(csv_paths, files_to_use):
    """
    Loads all specified CSV files, filters them by the selected images, 
    and returns a master dictionary for fast lookup.
    """
    master_data = {}
    
    # 1. Load and Filter each CSV into a Dictionary
    for source, path in csv_paths.items():
        if not os.path.exists(path):
            print(f"⚠️ Warning: CSV file not found at {path}. Skipping {source} model.")
            continue
            
        df = pd.read_csv(path)
        
        # Filter the DataFrame to only include the files we care about
        if files_to_use:
            df = df[df['filename'].isin(files_to_use)]
        
        # Convert DataFrame rows to a dictionary indexed by filename for fast lookup
        # { '00000.jpg': [xc, yc, w, h, conf], ... }
        data_dict = df.set_index('filename').T.to_dict('list')
        master_data[source] = data_dict

    if not master_data:
        print("FATAL: No label data could be loaded from any source.")
        sys.exit(1)
        
    # Get the list of filenames present across ALL loaded CSVs (union of files)
    # We use SELECTED_FILES list to maintain the display order
    available_files = []
    for f in files_to_use:
        # Check if the file is present in *at least one* loaded source
        if any(f in data_dict for data_dict in master_data.values()):
             available_files.append(f)

    if not available_files:
        print("FATAL: No data records match the SELECTED_FILES filter across any loaded CSVs.")
        sys.exit(1)
        
    return master_data, available_files


def draw_yolo_box(image, x_center, y_center, width, height, confidence, color, label_prefix):
    """
    Converts normalized YOLO box coordinates to pixel coordinates and draws 
    the box and confidence score on the image.
    """
    H, W = image.shape[:2]
    
    # 1. Convert normalized [x_c, y_c, w, h] to pixel [x_min, y_min, x_max, y_max]
    x_min = int((x_center - width / 2) * W)
    y_min = int((y_center - height / 2) * H)
    x_max = int((x_center + width / 2) * W)
    y_max = int((y_center + height / 2) * H)

    line_thickness = 2
    
    # 2. Draw the Bounding Box Rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, line_thickness)

    # 3. Draw the Label Text
    label = f"{label_prefix}: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    
    # Position text near the top left of the box
    text_x = x_min
    text_y = y_min - 5 if y_min > 15 else y_min + 20 

    # Draw the text
    cv2.putText(image, label, (text_x, text_y), font, font_scale, color, line_thickness, cv2.LINE_AA)


# --- 3. MAIN VISUALIZATION LOOP ---

def visualize_labels():
    """
    Loads labels from all sources and starts the interactive visualization loop.
    """
    # Load all label data, filtered by SELECTED_FILES
    all_labels_by_source, filenames_to_display = load_all_labels(LABEL_CSVS, SELECTED_FILES)
    
    total_images = len(filenames_to_display)
    current_index = 0
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    
    print("-" * 50)
    print(f"🖼️ Loaded {total_images} images for QA. Press Q/W to navigate.")
    print(f"Colors: Green (Teacher), Yellow (Student), Red (Random)")
    print("Visibility is controlled by the BOX_SETTINGS global dictionary.")
    print("-" * 50)
    
    # Initialize the current box settings (visibility flags)
    global BOX_SETTINGS
    current_box_settings = BOX_SETTINGS.copy() 
    
    while True:
        # Wrap index around the dataset size
        current_index = current_index % total_images
        
        filename = filenames_to_display[current_index]
        image_path = os.path.join(IMAGE_DIR, filename)
        
        # Load the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"⚠️ Image not found: {image_path}. Skipping.")
            current_index += 1
            continue

        # --- Draw Bounding Boxes from all sources ---
        y_offset = 0
        boxes_drawn = 0

        # Iterate through each model source and draw its prediction
        for source, (color, visible) in current_box_settings.items():
            if source in all_labels_by_source and filename in all_labels_by_source[source]:
                # Data is stored as a list: [x_c, y_c, w, h, confidence]
                data = all_labels_by_source[source][filename]
                
                # Check if the confidence score (index 4) is positive
                if data[4] > 0.0:
                    x_c, y_c, w, h, confidence = data
                    
                    if visible:
                        draw_yolo_box(image, x_c, y_c, w, h, confidence, color, source.capitalize())
                        boxes_drawn += 1
                
                elif data[4] == 0.0 and visible:
                     # This means the model explicitly predicted "No Surfer"
                     cv2.putText(image, f"{source.capitalize()}: NO SURFER DETECTED", 
                                 (10, 30 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                 0.6, color, 1, cv2.LINE_AA)
                     y_offset += 25
                elif data[4] == 0.0:
                    y_offset += 25 # Maintain spacing even if text is hidden


        if boxes_drawn == 0 and any(v for _, v in current_box_settings.values()):
            cv2.putText(image, "NO POSITIVE DETECTION ACROSS VISIBLE MODELS", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display index and filename in the console
        print(f"Displaying {current_index + 1}/{total_images}: {filename} (Visible Boxes: {boxes_drawn})")
        
        cv2.imshow(WINDOW_NAME, image)
        
        # Wait for a keypress
        key = cv2.waitKey(0) & 0xFF
        
        # Handle key presses
        if key == PREV_KEY: # 'q' key
            current_index -= 1
        elif key == NEXT_KEY: # 'w' key
            current_index += 1
        elif key == ESC_KEY: # ESC key to quit
            break

    cv2.destroyAllWindows()
    print("Label visualization finished.")

# --- Run the Script ---
if __name__ == "__main__":
    visualize_labels()
