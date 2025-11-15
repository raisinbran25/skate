import cv2
import numpy as np
import mss
import keyboard
import time
import os
import atexit
import slide_filenames

# --- Global Configuration ---
DIM = 416  # Target dimension (your CNN input size: DIM x DIM)
OUTPUT_FOLDERS = ["single_surfer_recognition/training_photos", "single_surfer_recognition/testing_photos"]
FPS_TARGET = 10 # Frames per second to capture
IMG_EXTENSION = "jpg"
# Keyboard keys for control
START_KEY = 'q'
STOP_KEY = 'w'

# --- Screen Region Configuration ---
# DEFINE THE REGION YOU WANT TO RECORD (in pixels, relative to the top-left of the screen)
# This is an example for a 800x600 region starting at (100, 50)
MONITOR_REGION = {
    "top": 224,     # Y coordinate of the top edge
    "left": 0,   # X coordinate of the left edge
    "width": 939,  # Width of the recording area
    "height": 536  # Height of the recording area
}

# --- Utility Functions (Same as before) ---

def letterbox_resize(frame, dim):
    """
    Resizes an image and pads it to a square (dim x dim) size with black bars.
    """
    h, w = frame.shape[:2]
    scale = dim / max(h, w)
    
    new_w = int(w * scale)
    new_h = int(h * scale)

    pad_w = dim - new_w
    pad_h = dim - new_h
    
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    letterboxed_frame = cv2.copyMakeBorder(
        resized_frame, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0] # Black bars
    )
    
    return letterboxed_frame

# --- Main Recording Logic ---

def interactive_screen_recorder(target_dim, fps_target, region, output_folders, img_extension, start_key, stop_key):
    """
    Starts and stops screen capture based on key presses.
    """
    # choose to collect training or testing photos
    def chooseFolder():
        while True:
            choice = input("enter 1 to collect training photos, and 2 for testing photos: ")
            if choice == "1":
                return output_folders[0]
            if choice == "2":
                return output_folders[1]
            
    output_folder = chooseFolder()

    os.makedirs(output_folder, exist_ok=True)

    # slide filenames
    slide_filenames.slide(output_folder)
    
    # Calculate initial frame counter based on existing files
    frame_counter = len([f for f in os.listdir(output_folder) if f.endswith(img_extension)])
    
    print("-" * 50)
    print("🌊 Surfer Detection Data Recorder")
    print(f"Target Size: {target_dim}x{target_dim} @ {fps_target} FPS")
    print(f"Recording Region: {region['width']}x{region['height']} at ({region['left']}, {region['top']})")
    print("-" * 50)
    print(f"Press **{start_key.upper()}** to **START** recording.")
    print(f"Press **{stop_key.upper()}** to **STOP** recording.")
    print(f"Press **CTRL+C** in the console to exit the script.")
    print("-" * 50)

    is_recording = False
    sct = mss.mss() # Initialize screen capture library
    
    # Time interval between captures to achieve target FPS
    target_interval = 1.0 / fps_target 

    # Key press handlers
    def start_recording_handler(e):
        nonlocal is_recording
        if not is_recording:
            is_recording = True
            print(f"⏺️ Recording STARTED. Frames will be saved to: {output_folder}")

    def stop_recording_handler(e):
        nonlocal is_recording
        if is_recording:
            is_recording = False
            print("⏹️ Recording STOPPED. Ready for next clip.")

    # Register hotkeys
    keyboard.on_press_key(start_key, start_recording_handler)
    keyboard.on_press_key(stop_key, stop_recording_handler)

    atexit.register(keyboard.unhook_all) # Ensure hotkeys are released on exit

    try:
        while True:
            if is_recording:
                start_time = time.time()

                # 1. Capture Screen Region using mss
                sct_img = sct.grab(region)
                # Convert the raw capture to a format readable by OpenCV/NumPy
                frame = np.array(sct_img)
                # Convert from BGRA (mss output) to BGR (OpenCV standard)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # 2. Process and Save Frame
                processed_frame = letterbox_resize(frame, target_dim)
                
                filename = os.path.join(
                    output_folder, 
                    f"{frame_counter:05d}.{img_extension}"
                )
                
                cv2.imwrite(filename, processed_frame)
                frame_counter += 1
                
                if frame_counter % (fps_target * 5) == 0: # Print update every 5 seconds
                    print(f"  -> Saved {frame_counter} total frames...")

                # 3. Control Frame Rate
                elapsed_time = time.time() - start_time
                sleep_time = target_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            else:
                # Sleep briefly when not recording to save CPU
                time.sleep(0.1) 
                
    except KeyboardInterrupt:
        print("\nScript manually stopped.")
    finally:
        print(f"\n✨ Screen recording session finished. Total frames saved: {frame_counter}")

# --- Run the Script ---
if __name__ == "__main__":
    interactive_screen_recorder(
        DIM, 
        FPS_TARGET, 
        MONITOR_REGION, 
        OUTPUT_FOLDERS, 
        IMG_EXTENSION, 
        START_KEY, 
        STOP_KEY
    )