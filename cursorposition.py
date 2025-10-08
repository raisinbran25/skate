import pyautogui
import time

try:
    while True:
        x, y = pyautogui.position()  # get cursor coordinates
        print(f"Cursor position: ({x}, {y})")
        time.sleep(2)  # wait 0.5 seconds
except KeyboardInterrupt:
    print("\nStopped by user.")
