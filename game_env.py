from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import pyautogui
import pygetwindow as gw
import pytesseract
import matplotlib.pyplot as plt
import mss

import time

from scripts import *

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def preprocess_templates_with_svd(templates, target_shape):
    """Process templates using SVD for improved matching."""
    processed_templates = []
    
    for template in templates:
        if template is None or template.size == 0:
            continue
            
        if template.shape[0] >= target_shape[0] or template.shape[1] >= target_shape[1]:
            scale = min(0.9 * target_shape[0] / template.shape[0], 
                         0.9 * target_shape[1] / template.shape[1])
            new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
            template = cv2.resize(template, new_size)
        
        try:
            u, s, vh = np.linalg.svd(template, full_matrices=False)
            
            k = min(10, len(s))
            reconstructed = u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]
            
            reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            processed_templates.append(reconstructed)
        except:
            processed_templates.append(template)
        
    return processed_templates

def get_bluestacks_coords():
    window = gw.getWindowsWithTitle("BlueStacks")[0]
    return window.left, window.top, window.width, window.height

bs_left, bs_top, bs_width, bs_height = get_bluestacks_coords()

def get_init_pic():
    with mss.mss() as sct:
        monitor = {"left": bs_left, "top": bs_top, "width": bs_width, "height": bs_height}
        screenshot = np.array(sct.grab(monitor))
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
        return gray

ss = get_init_pic()

ignore_multiplier = True

x_mult1 = int(0.5681818181818182 * ss.shape[1])
x_mult2 = int(0.7173295454545454 * ss.shape[1])
y_mult1 = int(0.023961661341853034 * ss.shape[0])
y_mult2 = int(0.06389776357827476 * ss.shape[0])

xST = int(0.7173295454545454 * ss.shape[1])
xED = int(1 * ss.shape[1])
yST = int(0.023961661341853034 * ss.shape[0])
yED = int(0.06389776357827476 * ss.shape[0])

x1 = int(0.4018547140649153 * ss.shape[1])
x2 = int(0.6336939721792893 * ss.shape[1])
y1 = int(0.09894459102902374 * ss.shape[0])
y2 = int(0.7915567282321899 * ss.shape[0])

print(ss.shape)

# In the initialization section, load full templates without cropping
template = cv2.imread("bluestacks_screenshot_gameover.png", cv2.IMREAD_GRAYSCALE)
template = np.array(template)
# Remove cropping
# template = template[int(y1*template.shape[0]/ss.shape[0]):int(y2*template.shape[0]/ss.shape[0]), int(x1*template.shape[1]/ss.shape[1]):int(x2*template.shape[1]/ss.shape[1])]

template2 = cv2.imread("bluestacks_screenshot_gameover2.png", cv2.IMREAD_GRAYSCALE)
template2 = np.array(template2)
# Remove cropping
# template2 = template2[int(y1*template2.shape[0]/ss.shape[0]):int(y2*template2.shape[0]/ss.shape[0]), int(x1*template2.shape[1]/ss.shape[1]):int(x2*template2.shape[1]/ss.shape[1])]

template3 = cv2.imread("bluestacks_screenshot_gameover3.png", cv2.IMREAD_GRAYSCALE)
template3 = np.array(template3)
# Remove cropping
# template3 = template3[int(y1*template3.shape[0]/ss.shape[0]):int(y2*template3.shape[0]/ss.shape[0]), int(x1*template3.shape[1]/ss.shape[1]):int(x2*template3.shape[1]/ss.shape[1])]

template4 = cv2.imread("bluestacks_screenshot_gameover4.png", cv2.IMREAD_GRAYSCALE)
template4 = np.array(template4)
# Remove cropping
# template3 = template3[int(y1*template3.shape[0]/ss.shape[0]):int(y2*template3.shape[0]/ss.shape[0]), int(x1*template3.shape[1]/ss.shape[1]):int(x2*template3.shape[1]/ss.shape[1])]
class SubwayEnv(gym.Env):

    def __init__(self, frame_stack=4, frame_size=(84, 84)):
        super(SubwayEnv, self).__init__()
        global ss

        self.action_space = spaces.Discrete(5)

        self.score = 0
        self.previous_score = 0
        self.steps = 0
        self.game_over = False
        
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        
        self.obs_shape = (frame_stack, *frame_size)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.obs_shape, dtype=np.uint8
        )

        self.frames = np.zeros(self.obs_shape, dtype=np.uint8)

        self.actions = {
            0: lambda: swipe_left_bluestacks(70),
            1: lambda: swipe_right_bluestacks(70),
            2: lambda: swipe_up_bluestacks(70),
            3: lambda: swipe_down_bluestacks(70),
            4: lambda: self.no_action()
        }

        self.rewards = {
            'game_over': -10,
            'survive': 0.1,
            'diff_multiplier': 0.5,
        }
        self.case = -1

        # Create persistent mss instance for faster screenshots
        self.sct = mss.mss()
        self.monitor = {"left": bs_left, "top": bs_top, "width": bs_width, "height": bs_height}
    
    def no_action(self):
        time.sleep(0.1)  # Remove delay
        print("---------No action performed")  # Comment out print
        return True
    
    def reset(self, **kwargs):
        self._restart_game()

        self.score = 0
        self.previous_score = 0
        self.steps = 0
        self.game_over = False

        self.frames = np.zeros(self.obs_shape, dtype=np.uint8)

        for i in range(self.frame_stack):
            self.frames[i] = self._get_observation()
            # time.sleep(0.1)  # Remove delay
        time.sleep(1.0)  # Remove delay
        self.score = self._extract_score()
        if self.score == 0 and hasattr(self, '_prev_reset_attempts'):
            self._prev_reset_attempts += 1
            if self._prev_reset_attempts < 3:
                print("Game may not have restarted properly. Trying again...")
                start_game()
            else: 
                print("waiting for user to do something")
                self._prev_reset_attempts = 0
                time.sleep(5)  # Keep this longer timeout for user intervention
                self.reset(**kwargs)
        else:
            self._prev_reset_attempts = 0
            
        return self.frames, {}
    
    def step(self, action):
        self.actions[action]()
        self.steps += 1
        self.game_over, self.case = self._detect_game_over()
        
        if self.game_over:
            # print("Game over detected!")  # Comment out print
            time.sleep(0.5)  # Remove delay
            return self.frames, -10, True, False, {"score": self.score, "steps": self.steps}

        # time.sleep(0.1)  # Remove delay
        new_frame = self._get_observation()

        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = new_frame

        reward = self._calculate_reward()

        return self.frames, reward, self.game_over, False, {"score": self.score, "steps": self.steps}
    
    def _get_observation(self):
        screenshot = self._capture_game_screen()

        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (self.frame_size[1], self.frame_size[0]), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def _capture_game_screen(self):
        screenshot = np.array(self.sct.grab(self.monitor))
        return screenshot
    
    def __del__(self):
        # Clean up resources
        if hasattr(self, 'sct'):
            self.sct.close()
    
    def _calculate_reward(self):
        # Comment out the original score-based reward system
        # self.score = self._extract_score()
        # reward = 0.1
        # score_diff = self.score - self.previous_score
        # if score_diff > 0:
        #     reward += self.rewards['diff_multiplier'] * score_diff

        # New reward system based on survival duration
        reward = 1.0  # Fixed reward per action performed

        # Keep the game over penalty
        if self.game_over:
            reward += self.rewards['game_over']  # -10 penalty on game over

        # Still update score for informational purposes only
        self.score = self._extract_score()
        self.previous_score = self.score
        
        return reward
    
    def _detect_game_over(self):
        # Initialize counter if not exists
        if not hasattr(self, '_game_over_counter'):
            self._game_over_counter = 0
        
        # Use current frame
        frame = self.frames[-1].copy()
        
        # Apply thresholding to improve OCR text detection
        _, thresh = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY_INV)

        # plt.imsave("./debug/thresh.png", thresh, cmap='gray')
        time.sleep(0.01)  # buffer time for the screenshot to stabilize
        # Perform OCR on the frame
        text = pytesseract.image_to_string(thresh, config='--psm 11')
        
        # Check if "SAVE" appears in the text (case insensitive)
        is_game_over = "SAVE" in text.upper()
        if is_game_over:
            self._game_over_counter += 1
            # print(f"Game over candidate detected via OCR! Text found: {text}")
        else:
            self._game_over_counter = max(0, self._game_over_counter - 1)
        
        # Only confirm game over after seeing it multiple times to avoid false positives
        confirmed_game_over = self._game_over_counter >= 1
        
        if confirmed_game_over:
            # print("Game over confirmed via OCR!")
            return True, 0  # Using case 0 for OCR detection
        
        return False, 2
            
    def _extract_score(self):
        global xST, xED, yST, yED

        np_img = self.frames[-1]
        cropped = np_img[yST:yED, xST:xED]
        
        # cv2.imwrite("./debug/score.png", cropped)  # Comment out image saving
        
        score_text = pytesseract.image_to_string(cropped, config='--psm 7 digits')
        
        digits_only = ''.join(c for c in score_text if c.isdigit())
        
        try:
            score = int(digits_only) if digits_only else 0
        except ValueError:
            # print(f"Warning: Could not parse score text: '{score_text}'")  # Comment out print
            score = self.previous_score
        
        if ignore_multiplier:
            try:
                return score / self._get_multiplier()
            except:
                return score
        
        return score
    
    def _get_multiplier(self):
        global x_mult1, x_mult2, y_mult1, y_mult2
        np_img = self.frames[-1]
        cropped = np_img[y_mult1:y_mult2, x_mult1:x_mult2]

        # cv2.imwrite("./debug/multiplier.png", cropped)  # Comment out image saving

        multiplier_text = pytesseract.image_to_string(cropped, config='--psm 7 digits')
        
        digits_only = ''.join(c for c in multiplier_text if c.isdigit())
        
        try:
            multiplier = int(digits_only) if digits_only else 1
            return max(1, multiplier)
        except ValueError:
            # print(f"Warning: Could not parse multiplier text: '{multiplier_text}'")  # Comment out print
            return 1
    
    def _restart_game(self):
        restart_game(self.case)


