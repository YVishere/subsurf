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
import threading
import queue

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
        self.previous_score = -10
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
        self.previous_score = -10
        self.steps = 0
        self.game_over = False

        self.frames = np.zeros(self.obs_shape, dtype=np.uint8)

        for i in range(self.frame_stack):
            self.frames[i], self.non_resized = self._get_observation()
            # time.sleep(0.1)  # Remove delay
        time.sleep(1.0)  # Remove delay
        self.score = self._extract_score()
        while self.score < 0 and hasattr(self, '_prev_reset_attempts') and self._detect_game_over():
            time.sleep(2)
            self.score = self._extract_score()
            self._prev_reset_attempts += 1
            if self._prev_reset_attempts < 8:
                print("Game may not have restarted properly. Trying again...")
                start_game()
                for i in range(self.frame_stack):
                    self.frames[i], self.non_resized = self._get_observation()
            else: 
                print("waiting for user to do something")
                self._prev_reset_attempts = 0
                time.sleep(5)  # Keep this longer timeout for user intervention
                return self.reset(**kwargs)
        
        self._prev_reset_attempts = 0
            
        return self.frames, {}
    
    def step(self, action):
        self.actions[action]()
        self.steps += 1
        self.game_over, self.case = self._detect_game_over()
        
        score_toret = self.score if self.score>=0 else self.previous_score
        
        if self.game_over:
            # print("Game over detected!")  # Comment out print
            # time.sleep(0.5)  # Remove delay
            return self.frames, -10, True, False, {"score": score_toret, "steps": self.steps}

        # time.sleep(0.1)  # Remove delay
        new_frame, self.non_resized = self._get_observation()

        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = new_frame

        reward = self._calculate_reward()

        return self.frames, reward, self.game_over, False, {"score": score_toret, "steps": self.steps}
    
    def _get_observation(self):
        screenshot1 = self._capture_game_screen()

        # Downsample first before converting to grayscale
        screenshot = cv2.resize(screenshot1, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Final resize to target shape
        resized = cv2.resize(gray, (self.frame_size[1], self.frame_size[0]), interpolation=cv2.INTER_AREA)
        return resized, screenshot1
    
    def _capture_game_screen(self):
        screenshot = np.array(self.sct.grab(self.monitor))
        return screenshot
    
    def __del__(self):
        # Clean up resources
        self.ocr_thread_running = False
        try:
            # Signal thread to exit
            self.ocr_queue.put(None)
            self.ocr_thread.join(timeout=1.0)
        except:
            pass
            
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
        self.previous_score = self.score if self.score >= 0 else self.previous_score
        
        return reward
    
    def _detect_game_over(self):
        print(f"Score: {self.score}")  # Comment out print
        if self.score < 0:
            return True, 0
        
        if self.score == 7 and self.previous_score == 7:
            return True, 0
        
        return False, 2
    
    def _extract_score(self):
        global xST, xED, yST, yED

        np_img = self.non_resized
        cropped = np_img[yST:yED, xST:xED]
        
        cv2.imwrite("./debug/score.png", cropped)  # Comment out image saving
        
        score_text = pytesseract.image_to_string(cropped, config='--psm 7 digits')
        
        digits_only = ''.join(c for c in score_text if c.isdigit())
        
        if digits_only == "":
            return -1

        try:
            score = int(digits_only) if digits_only else 0
        except ValueError:
            score = 0
        
        if ignore_multiplier:
            try:
                return score / self._get_multiplier()
            except:
                return score
        
        return score
    
    def _get_multiplier(self):
        global x_mult1, x_mult2, y_mult1, y_mult2
        np_img = self.non_resized
        cropped = np_img[y_mult1:y_mult2, x_mult1:x_mult2]

        # cv2.imwrite("./debug/multiplier.png", cropped)  # Comment out image saving

        multiplier_text = pytesseract.image_to_string(cropped, config='--psm 7 digits')
        
        digits_only = ''.join(c for c in multiplier_text if c.isdigit())
        
        try:
            multiplier = int(digits_only) if digits_only else 1
            return max(1, multiplier)
        except ValueError:
            return 1
    
    def _restart_game(self):
        restart_game(self.case)


