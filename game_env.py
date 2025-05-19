from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import pyautogui
import pygetwindow as gw
import pytesseract
import matplotlib.pyplot as plt

import time

from scripts import *

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Add these functions at the top of your file after imports
def preprocess_templates_with_svd(templates, target_shape):
    """Process templates using SVD for improved matching."""
    processed_templates = []
    
    for template in templates:
        if template is None or template.size == 0:
            continue
            
        # First resize template to be smaller than target region
        # Template must be smaller than the region we're searching
        if template.shape[0] >= target_shape[0] or template.shape[1] >= target_shape[1]:
            scale = min(0.9 * target_shape[0] / template.shape[0], 
                         0.9 * target_shape[1] / template.shape[1])
            new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
            template = cv2.resize(template, new_size)
        
        # Apply SVD for more robust matching
        try:
            u, s, vh = np.linalg.svd(template, full_matrices=False)
            
            # Keep only most significant components
            k = min(10, len(s))  # Keep top k singular values
            reconstructed = u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]
            
            # Normalize to 0-255 range
            reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            processed_templates.append(reconstructed)
        except:
            # Fallback if SVD fails
            processed_templates.append(template)
        
    return processed_templates

def get_init_pic():
    # Capture the screen using OpenCV
    window = gw.getWindowsWithTitle("BlueStacks")[0]
    window.activate()
    # Get window position and dimensions
    left, top, width, height = window.left, window.top, window.width, window.height
    
    # Capture screenshot
    screenshot = pyautogui.screenshot(region=(left, top, width, height))

    gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)

    # Convert to numpy array
    return np.array(gray)

ss = get_init_pic()

##################################
# Score multiplier ignore setting
##################################

#Score multipliers might mess up the reward system
# If you do not want to ignore them, set this to False
ignore_multiplier = True

x_mult1 = int(0.5681818181818182 * ss.shape[1])
x_mult2 = int(0.7173295454545454 * ss.shape[1])
y_mult1 = int(0.023961661341853034 * ss.shape[0])
y_mult2 = int(0.06389776357827476 * ss.shape[0])

# Dimensions of score box
xST = int(0.7173295454545454 * ss.shape[1])
xED = int(1 * ss.shape[1])
yST = int(0.023961661341853034 * ss.shape[0])
yED = int(0.06389776357827476 * ss.shape[0])

# Game end masks
x1 = int(0.4018547140649153 * ss.shape[1])
x2 = int(0.6336939721792893 * ss.shape[1])
y1 = int(0.09894459102902374 * ss.shape[0])
y2 = int(0.7915567282321899 * ss.shape[0])

print(ss.shape)

template = cv2.imread("bluestacks_screenshot_gameover.png", cv2.IMREAD_GRAYSCALE)
template = np.array(template)
template = template[int(y1*template.shape[0]/ss.shape[0]):int(y2*template.shape[0]/ss.shape[0]), int(x1*template.shape[1]/ss.shape[1]):int(x2*template.shape[1]/ss.shape[1])]

template2 = cv2.imread("bluestacks_screenshot_gameover2.png", cv2.IMREAD_GRAYSCALE)
template2 = np.array(template2)
template2 = template2[int(y1*template2.shape[0]/ss.shape[0]):int(y2*template2.shape[0]/ss.shape[0]), int(x1*template2.shape[1]/ss.shape[1]):int(x2*template2.shape[1]/ss.shape[1])]

template3 = cv2.imread("bluestacks_screenshot_gameover3.png", cv2.IMREAD_GRAYSCALE)
template3 = np.array(template3)
template3 = template3[int(y1*template3.shape[0]/ss.shape[0]):int(y2*template3.shape[0]/ss.shape[0]), int(x1*template3.shape[1]/ss.shape[1]):int(x2*template3.shape[1]/ss.shape[1])]
class SubwayEnv(gym.Env):

    def __init__(self, frame_stack=4, frame_size=(84, 84)):
        super(SubwayEnv, self).__init__()
        global ss

        self.action_space = spaces.Discrete(5)  # 5 actions: left, right, up, down, none

        # Game state tracking
        self.score = 0
        self.previous_score = 0
        self.steps = 0
        self.game_over = False
        
        # FIX: Use the parameter value instead of screenshot height
        self.frame_stack = frame_stack  # Changed from ss.shape[0]
        self.frame_size = frame_size  # Use the input parameter which is already (84, 84)
        
        # Fix observation shape dimensions
        # Using grayscale, so channels = 1
        self.obs_shape = (frame_stack, *frame_size)  # Results in (4, 84, 84)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.obs_shape, dtype=np.uint8
        )

        self.frames = np.zeros(self.obs_shape, dtype=np.uint8)

        self.actions = {
            0: lambda: swipe_left_bluestacks(70),
            1: lambda: swipe_right_bluestacks(70),
            2: lambda: swipe_up_bluestacks(70),
            3: lambda: swipe_down_bluestacks(70),
            4: lambda: time.sleep(0.1)  # No action
        }

        self.rewards = {
            'game_over': -10,
            'survive': 0.1,
            'diff_multiplier': 0.5,
        }
        self.case = -1
    
    def reset(self, **kwargs):
        self._restart_game()

        self.score = 0
        self.previous_score = 0
        self.steps = 0
        self.game_over = False

        self.frames = np.zeros(self.obs_shape, dtype=np.uint8)

        for i in range(self.frame_stack):
            self.frames[i] = self._get_observation()
            time.sleep(0.1)
        time.sleep(1.0)
        self.score = self._extract_score()
        if self.score == 0 and hasattr(self, '_prev_reset_attempts'):
            self._prev_reset_attempts += 1
            if self._prev_reset_attempts < 3:
                print("Game may not have restarted properly. Trying again...")
                start_game()
            else: 
                print("waiting for user to do something")
                self._prev_reset_attempts = 0
                time.sleep(5)
                self.reset(**kwargs)
        else:
            self._prev_reset_attempts = 0
            
        return self.frames, {}
    
    def step(self, action):
        self.actions[action]()
        self.steps += 1
        self.game_over, self.case = self._detect_game_over()
        
        if self.game_over:
            print("Game over detected!")
            # Add a short pause to stabilize
            time.sleep(0.5)
            return self.frames, -10, True, False, {"score": self.score, "steps": self.steps}

        time.sleep(0.1)
        new_frame = self._get_observation()

        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = new_frame

        reward = self._calculate_reward()

        return self.frames, reward, self.game_over, False, {"score": self.score, "steps": self.steps}
    
    def _get_observation(self):
        # Capture the screen and process the image
        screenshot = self._capture_game_screen()

        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Resize to frame_size (width, height)
        resized = cv2.resize(gray, (self.frame_size[1], self.frame_size[0]), interpolation=cv2.INTER_AREA)
        
        # NOTE: cv2.resize expects (width, height) but NumPy arrays are (height, width)
        # No need to transpose if dimensions are already correct
        
        return resized
    
    def _capture_game_screen(self):
        # Capture the screen using OpenCV
        window = gw.getWindowsWithTitle("BlueStacks")[0]
        window.activate()
        # Get window position and dimensions
        left, top, width, height = window.left, window.top, window.width, window.height
        
        # Capture screenshot
        screenshot = pyautogui.screenshot(region=(left, top, width, height))

        screenshot.save("./debug/screenshot.png")

        # Convert to numpy array
        return np.array(screenshot)
    
    def _calculate_reward(self):
        self.score = self._extract_score()

        reward = 0.1

        score_diff = self.score - self.previous_score
        if score_diff > 0:
            reward += self.rewards['diff_multiplier'] * score_diff

        if self.game_over:
            reward += self.rewards['game_over']

        self.previous_score = self.score
        return reward
    
    def _detect_game_over(self):
        global template, template2, template3
        global x1, x2, y1, y2
        
        # Initialize game_over counter if not present
        if not hasattr(self, '_game_over_counter'):
            self._game_over_counter = 0
            # Fix template2 initialization if needed
            if np.array_equal(template, template2):
                template2 = cv2.imread("bluestacks_screenshot_gameover2.png", cv2.IMREAD_GRAYSCALE)
                if template2 is not None:
                    template2 = template2[int(y1*template2.shape[0]/ss.shape[0]):int(y2*template2.shape[0]/ss.shape[0]), 
                                         int(x1*template2.shape[1]/ss.shape[1]):int(x2*template2.shape[1]/ss.shape[1])]
            
            # Create processed templates scaled to match ROI
            self.templates = [template, template2, template3] if template2 is not None else [template]
        
        # Extract ROI from current frame
        try:
            gray = self.frames[-1][y1:y2, x1:x2].copy()
            cv2.imwrite("./debug/game_over_roi.png", gray)
        except:
            print("Error extracting ROI, using full frame")
            gray = self.frames[-1].copy()
        
        # Process templates if not done yet
        if not hasattr(self, 'processed_templates'):
            self.processed_templates = preprocess_templates_with_svd(self.templates, gray.shape)
        
        # Check each template
        is_game_over = False
        match_index = -1
        best_confidence = 0
        
        for i, proc_template in enumerate(self.processed_templates):
            try:
                # Ensure template is smaller than ROI
                if proc_template.shape[0] > gray.shape[0] or proc_template.shape[1] > gray.shape[1]:
                    scale = min(0.9 * gray.shape[0] / proc_template.shape[0], 
                                0.9 * gray.shape[1] / proc_template.shape[1])
                    new_size = (int(proc_template.shape[1] * scale), int(proc_template.shape[0] * scale))
                    proc_template = cv2.resize(proc_template, new_size)
                
                # Try template matching
                result = cv2.matchTemplate(gray, proc_template, cv2.TM_CCOEFF_NORMED)
                confidence = np.max(result)
                print(f"Template {i} matching result: {confidence:.4f}")
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    match_index = i
                    
                if confidence > 0.7:  # Slightly reduced threshold for robustness
                    is_game_over = True
            except Exception as e:
                print(f"Error matching template {i}: {e}")
        
        # Use temporal consistency for more robust detection
        if is_game_over:
            self._game_over_counter += 1
            print(f"Game over candidate detected! (Template #{match_index}, confidence: {best_confidence:.3f})")
        else:
            self._game_over_counter = max(0, self._game_over_counter - 1)
        
        # Require 2 consecutive detections to confirm game over
        confirmed_game_over = self._game_over_counter >= 2
        
        if confirmed_game_over:
            print(f"Game over confirmed! (Template #{match_index})")
            return True, match_index
        
        print("No game over detected.")
        return False, 2
            
    
    def _extract_score(self):
        global xST, xED, yST, yED

        np_img = self.frames[-1]
        cropped = np_img[yST:yED, xST:xED]
        
        # Save using OpenCV instead of matplotlib for better image quality
        cv2.imwrite("./debug/score.png", cropped)
        
        score_text = pytesseract.image_to_string(cropped, config='--psm 7 digits')
        
        # Clean the string to keep only digits
        digits_only = ''.join(c for c in score_text if c.isdigit())
        
        try:
            # Convert to integer if digits are found
            score = int(digits_only) if digits_only else 0
        except ValueError:
            # Fallback if conversion fails
            print(f"Warning: Could not parse score text: '{score_text}'")
            score = self.previous_score  # Use previous score as fallback
        
        if ignore_multiplier:
            try:
                return score / self._get_multiplier()
            except:
                return score  # Fallback if multiplier extraction fails
        
        return score
    
    def _get_multiplier(self):
        global x_mult1, x_mult2, y_mult1, y_mult2
        np_img = self.frames[-1]
        cropped = np_img[y_mult1:y_mult2, x_mult1:x_mult2]

        # Use OpenCV instead of matplotlib
        cv2.imwrite("./debug/multiplier.png", cropped)

        multiplier_text = pytesseract.image_to_string(cropped, config='--psm 7 digits')
        
        # Clean the string to keep only digits
        digits_only = ''.join(c for c in multiplier_text if c.isdigit())
        
        try:
            # Default to 1 if no multiplier found (prevents division by zero)
            multiplier = int(digits_only) if digits_only else 1
            # Multiplier can't be zero (would cause division by zero)
            return max(1, multiplier)
        except ValueError:
            print(f"Warning: Could not parse multiplier text: '{multiplier_text}'")
            return 1  # Default multiplier
    
    def _restart_game(self):
        restart_game(self.case)


