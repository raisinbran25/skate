import cv2
import numpy as np
import mss

region = {"top": 480, "left": 0, "width": 930, "height": 350}
sct = mss.mss()

while True:
    # Grab the screen region
    screenshot = sct.grab(region)
    frame = np.array(screenshot)[:, :, :3]

    # --- Detect black pixels ---
    # Define a threshold: anything <= (30,30,30) counts as "black"
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([30, 30, 30], dtype=np.uint8)
    mask = cv2.inRange(frame, lower, upper)  # white where black pixels are

    # Show both original frame and mask
    cv2.imshow("Screen Capture", frame)
    cv2.imshow("Black Pixels", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
