import os
import cv2
import mss
import csv
import numpy as np
from pathlib import Path
import re

# --------- YOUR SETTINGS ---------
BBOX = {'left': 206, 'top': 227, 'width': 512, 'height': 512}

# "training" and "testing" to push images and bounding boxes to respective folders and csv
IMG_DIR = Path("./surfer/surf_training_images")   # output image folder
CSV_PATH = Path("./surfer/surf_training_data.csv")         # output CSV file


WINDOW   = "Surfer Label Tool"

# --------- GLOBAL STATE ---------
drawing = False
ix, iy = -1, -1
rects = []          # list of (x1, y1, x2, y2) in 512x512 coords
img = None          # image with drawings
img_base = None     # raw captured image

# --------- UTILITIES ---------
def ensure_outputs():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "cx", "cy", "w", "h"])  # header

def current_max_index():
    """
    Return the highest numeric stem found in IMG_DIR (e.g., 0007.jpg -> 7).
    If none found, return -1 so the next becomes 0 (0000.jpg).
    """
    pat = re.compile(r"^(\d{4})\.jpg$", re.IGNORECASE)
    max_idx = -1
    if not IMG_DIR.exists():
        return -1
    for p in IMG_DIR.iterdir():
        if p.is_file():
            m = pat.match(p.name)
            if m:
                try:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    pass
    return max_idx

def next_filename():
    idx = current_max_index() + 1
    return f"{idx:04d}.jpg"

def grab_screen():
    with mss.mss() as sct:
        arr = np.array(sct.grab(BBOX))  # BGRA
        frame = arr[:, :, :3]           # drop alpha -> BGR
    return frame

def clamp_box(x1, y1, x2, y2, W=512, H=512):
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    return x1, y1, x2, y2

def to_center_wh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h

# --------- MOUSE CALLBACK ---------
def on_mouse(event, x, y, flags, param):
    global drawing, ix, iy, img, img_base, rects
    if img_base is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x1, y1, x2, y2 = clamp_box(ix, iy, x, y)
        tmp = img_base.copy()
        cv2.rectangle(tmp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (rx1, ry1, rx2, ry2) in rects:
            cv2.rectangle(tmp, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        img = tmp

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1, x2, y2 = clamp_box(ix, iy, x, y)
        if abs(x2 - x1) >= 2 and abs(y2 - y1) >= 2:
            rects.append((x1, y1, x2, y2))
        img = img_base.copy()
        for (rx1, ry1, rx2, ry2) in rects:
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

# --------- SAVE ---------
def save_current():
    global img_base, rects
    if img_base is None:
        print("Nothing to save. Press 's' to capture first.")
        return

    fname = next_filename()
    out_path = IMG_DIR / fname

    # 1) Save image
    cv2.imwrite(str(out_path), img_base)

    # 2) Append rows to CSV (one per box)
    wrote = 0
    if rects:
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            for (x1, y1, x2, y2) in rects:
                cx, cy, bw, bh = to_center_wh(x1, y1, x2, y2)
                cx /= 512; cy /= 512; bw /= 512; bh /= 512 
                w.writerow([fname, f"{cx:.2f}", f"{cy:.2f}", f"{bw:.2f}", f"{bh:.2f}"])
                wrote += 1

    print(f"Saved {fname} ({wrote} box{'es' if wrote != 1 else ''}) -> {CSV_PATH.name}")
    rects = []

# --------- MAIN ---------
def main():
    global img, img_base, rects
    ensure_outputs()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("\nControls:")
    print("  s : capture the 512x512 screen crop")
    print("  left-click + drag : draw a box")
    print("  u : undo last box")
    print("  c : clear all boxes")
    print("  n : save image + boxes to disk/CSV")
    print("  q : (optionally save) and quit\n")

    blank = np.full((512, 512, 3), 30, dtype=np.uint8)
    img = blank.copy()
    img_base = None

    while True:
        cv2.imshow(WINDOW, img)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('s'):
            img_base = grab_screen()
            img = img_base.copy()
            rects = []

        elif k == ord('u'):
            if rects:
                rects.pop()
                img = img_base.copy() if img_base is not None else img
                for (x1, y1, x2, y2) in rects:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        elif k == ord('c'):
            rects = []
            if img_base is not None:
                img = img_base.copy()

        elif k == ord('n'):
            save_current()
            if img_base is None:
                img = blank.copy()

        elif k == ord('q'):
            if img_base is not None and rects:
                save_current()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
