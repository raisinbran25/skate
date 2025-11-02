import pandas as pd
import cv2
import os
import sys

# --- Configuration (Must Match Label Generator) ---
IMAGE_DIR = "single_surfer_recognition/training_photos"
LABEL_CSV = "single_surfer_recognition/teacher_labels.csv"
WINDOW_NAME = "Bounding Box QA Tool"

# Keys for navigation
NEXT_KEY = ord('w')
PREV_KEY = ord('q')

def draw_yolo_box(image, x_center, y_center, width, height, confidence):
    """
    Converts normalized YOLO box coordinates to pixel coordinates and draws 
    the box and confidence score on the image.
    """
    H, W = image.shape[:2]
    
    # 1. Convert normalized [x_c, y_c, w, h] to pixel [x_min, y_min, x_max, y_max]
    # x_min = (x_c - w/2) * W
    x_min = int((x_center - width / 2) * W)
    y_min = int((y_center - height / 2) * H)
    x_max = int((x_center + width / 2) * W)
    y_max = int((y_center + height / 2) * H)

    # 2. Define drawing parameters
    box_color = (0, 255, 0) # Green
    text_color = (255, 255, 255) # White
    line_thickness = 2
    
    # 3. Draw the Bounding Box Rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, line_thickness)

    # 4. Draw the Confidence Score
    label = f"Surfer: {confidence:.2f}"
    
    # Choose font and location for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_size = cv2.getTextSize(label, font, font_scale, line_thickness)[0]
    
    # Position text slightly above the box
    text_x = x_min
    text_y = y_min - 10 if y_min > 20 else y_max + 20 

    # Draw background box for text clarity
    cv2.rectangle(image, (text_x, text_y - text_size[1]), 
                  (text_x + text_size[0], text_y + 5), box_color, cv2.FILLED)
    
    # Draw the text
    cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, line_thickness, cv2.LINE_AA)

def visualize_labels():
    """
    Loads labels and images, then loops through them for interactive visualization.
    """
    try:
        # Load the CSV file containing the labels
        df = pd.read_csv(LABEL_CSV)
    except FileNotFoundError:
        print(f"❌ Error: Label file not found at {LABEL_CSV}. Please run the label generator script first.")
        sys.exit(1)
    
    total_images = len(df)
    current_index = 0
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    
    print("-" * 50)
    print(f"🖼️ Loaded {total_images} images for QA. Press Q/W to navigate.")
    print("-" * 50)
    
    while True:
        # Wrap index around the dataset size
        if current_index >= total_images:
            current_index = 0
        elif current_index < 0:
            current_index = total_images - 1

        row = df.iloc[current_index]
        filename = row['filename']
        image_path = os.path.join(IMAGE_DIR, filename)
        
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"⚠️ Image not found: {image_path}. Skipping.")
            current_index += 1
            continue

        # Check for positive detection (confidence > 0)
        confidence = row['confidence']
        if confidence > 0.0:
            # Extract normalized coordinates
            x_c = row['x_center']
            y_c = row['y_center']
            w = row['width']
            h = row['height']
            
            # Draw the bounding box on the image
            draw_yolo_box(image, x_c, y_c, w, h, confidence)
        else:
            # If no box was detected, display a message
            cv2.putText(image, "NO SURFER DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display index and filename in the console
        print(f"Displaying {current_index + 1}/{total_images}: {filename}")
        
        # Display the image
        cv2.imshow(WINDOW_NAME, image)
        
        # Wait for a keypress
        key = cv2.waitKey(0) & 0xFF
        
        # Handle key presses
        if key == PREV_KEY: # 'q' key
            current_index -= 1
        elif key == NEXT_KEY: # 'w' key
            current_index += 1
        elif key == 27: # ESC key to quit
            break

    cv2.destroyAllWindows()
    print("Label visualization finished.")

# --- Run the Script ---
if __name__ == "__main__":
    visualize_labels()