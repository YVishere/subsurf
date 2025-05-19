import pygetwindow as gw
import pyautogui
from PIL import Image
import time
import cv2
import numpy as np

def _detect_game_over( img):
        template = cv2.imread("bluestacks_screenshot_gameover.png", 0)
        template = np.array(template)[x1:x2, y1:y2]
        img = img[x1:x2, y1:y2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

        return np.max(result) > 0.8

def capture_bluestacks_screenshot():
    bluestacks_windows = gw.getWindowsWithTitle('BlueStacks')

    if not bluestacks_windows:
        print("BlueStacks window not found")
    else:
        bluestacks_window = bluestacks_windows[0]
        
        try:
            bluestacks_window.activate()
        except Exception as e:
            if "Error code from Windows: 0" in str(e):
                print("Window activation reported an error but seems to have succeeded")
            else:
                print(f"Window activation failed: {e}")
        
        left, top, width, height = bluestacks_window.left, bluestacks_window.top, bluestacks_window.width, bluestacks_window.height
        
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        
        return screenshot, screenshot.size

while True:
    ss, shape = capture_bluestacks_screenshot()
        
    x1 = int(0.4018547140649153 * shape[1])
    x2 = int(0.6336939721792893 * shape[1])
    y1 = int(0.09894459102902374 * shape[0])
    y2 = int(0.7915567282321899 * shape[0])

    np_img = np.array(ss)