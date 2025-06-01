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
template = cv2.imread("./reference_pics/bluestacks_screenshot_gameover.png", cv2.IMREAD_GRAYSCALE)
template = np.array(template)
# Remove cropping
# template = template[int(y1*template.shape[0]/ss.shape[0]):int(y2*template.shape[0]/ss.shape[0]), int(x1*template.shape[1]/ss.shape[1]):int(x2*template.shape[1]/ss.shape[1])]

template2 = cv2.imread("./reference_pics/bluestacks_screenshot_gameover2.png", cv2.IMREAD_GRAYSCALE)
template2 = np.array(template2)
# Remove cropping
# template2 = template2[int(y1*template2.shape[0]/ss.shape[0]):int(y2*template2.shape[0]/ss.shape[0]), int(x1*template2.shape[1]/ss.shape[1]):int(x2*template2.shape[1]/ss.shape[1])]

template3 = cv2.imread("./reference_pics/bluestacks_screenshot_gameover3.png", cv2.IMREAD_GRAYSCALE)
template3 = np.array(template3)
# Remove cropping
# template3 = template3[int(y1*template3.shape[0]/ss.shape[0]):int(y2*template3.shape[0]/ss.shape[0]), int(x1*template3.shape[1]/ss.shape[1]):int(x2*template3.shape[1]/ss.shape[1])]

template4 = cv2.imread("./reference_pics/bluestacks_screenshot_gameover4.png", cv2.IMREAD_GRAYSCALE)
template4 = np.array(template4)
# Remove cropping
# template3 = template3[int(y1*template3.shape[0]/ss.shape[0]):int(y2*template3.shape[0]/ss.shape[0]), int(x1*template3.shape[1]/ss.shape[1]):int(x2*template3.shape[1]/ss.shape[1])]
class SubwayEnv(gym.Env):

    def __init__(self, frame_stack=4, frame_size=(84, 84), frame_skip=2):
        super(SubwayEnv, self).__init__()
        global ss

        self.template_index = -1

        self.action_space = spaces.Discrete(5)

        self.score = 0
        self.previous_score = -10
        self.steps = 0
        self.game_over = False
        
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frame_skip = frame_skip
        
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
            'game_over': -150,
            'survive': 0.1,
            'diff_multiplier': 0.5,
        }
        self.case = -1

        # Create persistent mss instance for faster screenshots
        self.sct = mss.mss()
        self.monitor = {"left": bs_left, "top": bs_top, "width": bs_width, "height": bs_height}
        
        # Add score region change detection variables
        self.prev_score_region = None
        self.score_diff_threshold = 40  # Threshold for detecting significant changes
        self.score_change_counter = 0
        self.score_change_history = []
        
        # Setup screenshot buffer with threading
        self.screenshot_buffer = queue.Queue(maxsize=2)
        self.capture_thread_active = True
        self.capture_thread = threading.Thread(target=self._screenshot_worker, daemon=True)
        self.capture_thread.start()
            
        # Initialize reset attempts counter
        self._prev_reset_attempts = 0
        self.reset_delay = 1.5  # Seconds to wait between restart attempts
    
    def no_action(self):
        # Remove the delay and print statement
        print("---------No action taken")  # Comment out print
        return True
    
    def reset(self, **kwargs):
        # Clear game over state FIRST
        self.game_over = False
        self.case = -1
    
        # Ensure proper restart
        self._restart_game()
    
        # Reset all state variables
        self.score = 0
        self.previous_score = -10
        self.steps = 0
        self.prev_score_region = None
        self.score_change_counter = 0
    
        # Force clear frame buffer
        self.frames = np.zeros(self.obs_shape, dtype=np.uint8)
    
        # Clear screenshot buffer
        while not self.screenshot_buffer.empty():
            try:
                self.screenshot_buffer.get_nowait()
            except queue.Empty:
                break
    
        # Wait longer before interacting
        time.sleep(1.0)
    
        # Get initial observations
        for i in range(self.frame_stack):
            self.frames[i], self.non_resized = self._get_observation()
    
        # Longer wait for game to fully load
        time.sleep(2.5)  # Increased from 2.0
    
        # Start game with more reliable clicks
        for _ in range(2):  # Try multiple clicks
            start_game()
            time.sleep(0.5)
    
        # Refresh observation after game start
        for i in range(self.frame_stack):
            self.frames[i], self.non_resized = self._get_observation()

        # More robust restart verification
        self.score = self._extract_score()
        restart_attempts = 0
    
        # Add explicit validation that game is in playable state
        while (self.score < 0 or self._check_for_game_over_screen()) and restart_attempts < 5:
            print(f"Game may not have restarted properly. Trying again... (Attempt {restart_attempts+1}/5)")
            restart_attempts += 1
        
            # More forceful restart
            restart_game()  # Assume you add a force parameter to restart_game
            time.sleep(self.reset_delay * 1.5)  # Wait longer
        
            start_game()
            time.sleep(1.0)  # Wait longer after clicking
        
            # Refresh observations
            for i in range(self.frame_stack):
                self.frames[i], self.non_resized = self._get_observation()
                
            self.score = self._extract_score()
    
        print(f"Game restarted with score: {self.score}")
        return self.frames, {}
    
    def step(self, action):
        # Early return if already in game over state
        if self.game_over:
            print("Warning: step() called while game is already over")
            return self.frames, self.rewards['game_over'], True, False, {"score": 0, "steps": self.steps}
        
        reward_sum = 0
        done = False
        
        # Execute action immediately based on current state
        if not self.game_over:
            self.actions[action]()
        
        # Tiny wait for action to take effect in game
        time.sleep(0.01)
        
        # Main frame skip loop - now captures consequences of the action
        for i in range(self.frame_skip):
            # Get fresh observation to see results of action
            new_frame, self.non_resized = self._get_observation()
            
            # Update frame stack with this new observation
            self.frames = np.roll(self.frames, -1, axis=0)
            self.frames[-1] = new_frame
            
            # Check if game ended as a result of our action
            self.game_over, self.case = self._detect_game_over()
            
            # Calculate reward based on current state
            immediate_reward = self._calculate_reward() if not self.game_over else self.rewards['game_over']
            reward_sum += immediate_reward
            
            # Break out of frame skip if game ends
            if self.game_over:
                done = True
                break
                
            # Only skip remaining frames if not on last iteration
            if i < self.frame_skip - 1:
                time.sleep(0.01)  # Small wait between skipped frames
    
        self.steps += 1
        
        # Score updates (less frequent to reduce OCR load)
        if self.steps % 10 == 0 and not done:
            self.score = self._extract_score()
            self.previous_score = self.score if self.score >= 0 else self.previous_score
    
        score_toret = self.score if self.score >= 0 else self.previous_score
    
        return self.frames, reward_sum, done, False, {"score": score_toret, "steps": self.steps}
    
    def _get_observation(self):
        screenshot1 = self._capture_game_screen()

        # Downsample first before converting to grayscale
        screenshot = cv2.resize(screenshot1, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Final resize to target shape
        resized = cv2.resize(gray, (self.frame_size[1], self.frame_size[0]), interpolation=cv2.INTER_AREA)
        return resized, screenshot1
    
    def _screenshot_worker(self):
        """Background thread that captures screenshots"""
        # Create a separate mss instance specifically for this thread
        with mss.mss() as thread_sct:
            while self.capture_thread_active:
                try:
                    # Use the thread's own mss instance
                    screenshot = np.array(thread_sct.grab(self.monitor))
                    
                    # Add timestamp to track frame freshness
                    timestamp = time.time()
                    
                    # Clear queue before adding new frame to always have latest
                    while not self.screenshot_buffer.empty():
                        try:
                            self.screenshot_buffer.get_nowait()
                        except queue.Empty:
                            break
                    
                    # Put new frame
                    self.screenshot_buffer.put((screenshot, timestamp), block=False)
                    
                    # plt.imsave("./debug/screenshot.png", screenshot)  # Save for debugging
                    # Minimal sleep to prevent CPU overuse
                    time.sleep(0.001)  # Reduced from 0.005
                except Exception as e:
                    print(f"Screenshot worker error: {e}")
                    time.sleep(0.01)

    def _capture_game_screen(self):
        """Get latest screenshot from buffer with freshness check"""
        try:
            # Get screenshot and timestamp
            screenshot, timestamp = self.screenshot_buffer.get(block=False)
            
            # Check if screenshot is too old (more than 100ms)
            if time.time() - timestamp > 0.1:
                print("Warning: Using outdated frame")
                
            return screenshot
        except queue.Empty:
            # Fallback to direct capture
            return np.array(self.sct.grab(self.monitor))
    
    def __del__(self):
        # Stop capture thread
        self.capture_thread_active = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        
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
        self.score = self._extract_score()
        reward = 0.1
        score_diff = self.score - self.previous_score
        if score_diff > 0:
            reward += self.rewards['diff_multiplier'] * score_diff

        # New reward system based on survival duration
        # reward = 1.0  # Fixed reward per action performed

        # Keep the game over penalty
        if self.game_over:
            reward += self.rewards['game_over']  # -10 penalty on game over

        # Still update score for informational purposes only
        # self.score = self._extract_score()
        self.previous_score = self.score if self.score >= 0 else self.previous_score
        
        return reward
    
    def _detect_game_over(self):
        """Improved game over detection with early game protection"""
        # # Skip detection during the first few steps of an episode
        # if self.steps < 10:  # Don't detect game over too early
        #     return False, 2
        
        # Only use template matching, which is more reliable
        if self._check_templates_for_game_over():
            print(f"Game over detected by template at step {self.steps}, score {self.score}")
            return True, 0
        
        # Score region change detection has too many false positives - disable it
        # or make it extremely conservative
        global xST, xED, yST, yED
        
        # Extract score region with more padding
        padding = 15
        y_start = max(0, yST-padding)
        y_end = min(self.non_resized.shape[0], yED+padding*2)
        x_start = max(0, xST-padding)
        x_end = min(self.non_resized.shape[1], xED+padding)
        
        current_score_region = self.non_resized[y_start:y_end, x_start:x_end]
        
        # Convert to grayscale and use blur to reduce noise
        gray_region = cv2.cvtColor(current_score_region, cv2.COLOR_BGR2GRAY)
        gray_region = cv2.GaussianBlur(gray_region, (3, 3), 0)
        
        # Initialize reference on first run
        if self.prev_score_region is None:
            self.prev_score_region = gray_region
            return False, 2
        
        # Update reference with blend for stability
        self.prev_score_region = cv2.addWeighted(self.prev_score_region, 0.7, gray_region, 0.3, 0)
        
        # Only use score checks as last resort, and only if score is very low after many steps
        if self.score <= 0 and self.steps > 30:
            print(f"Game over detected: Invalid score {self.score} after {self.steps} steps")
            return True, 0
        
        return False, 2
    
    def _check_templates_for_game_over(self):
        """Check for game over screen using template matching"""
        global template, template2, template3, template4
        
        # Get smaller version of screenshot for faster processing
        screenshot = cv2.cvtColor(self.non_resized, cv2.COLOR_BGR2GRAY)
        small_screen = cv2.resize(screenshot, (0, 0), fx=0.3, fy=0.3)
        
        templates = [template, template2, template3, template4]
        self.template_index = -1
        for tmpl in templates:
            self.template_index  += 1
            if tmpl is None or tmpl.size == 0:
                continue
                
            # Ensure template is smaller than screenshot
            scale = min(0.25, 
                      (small_screen.shape[0]-10) / max(10, tmpl.shape[0]), 
                      (small_screen.shape[1]-10) / max(10, tmpl.shape[1]))
            
            try:
                small_template = cv2.resize(tmpl, (0, 0), fx=scale, fy=scale)
                
                # Only match if template is smaller than screen
                if (small_template.shape[0] < small_screen.shape[0] and 
                    small_template.shape[1] < small_screen.shape[1]):
                    
                    result = cv2.matchTemplate(small_screen, small_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # Higher threshold for more confidence
                    # print(max_val)
                    if max_val > 0.45:
                        print(f"Game over detected by template matching: {max_val:.2f}")
                        return True
            except Exception as e:
                continue
        
        return False
    
    def _extract_score(self):
        global xST, xED, yST, yED

        np_img = self.non_resized
        cropped = np_img[yST:yED, xST:xED]
        
        # cv2.imwrite("./debug/score.png", cropped)  # Comment out image saving
        
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
        """Improved restart with better timing"""
        if self.template_index == 1:
            start_game()
        else:
            restart_game(self.case)
        
        # Wait a moment for restart animation
        time.sleep(1.0)
        
        # Ensure window is active
        window = get_bluestacks_window()
        if window and not self.template_index == 1:
            # Additional click to ensure game starts
            x, y = to_absolute_coords(window, option=1)
            pyautogui.click(x, y)
            
        time.sleep(0.5)
    
    def _check_for_game_over_screen(self):
        """Dedicated method to check if current screen shows game over"""
        # Always use template matching for this specific purpose
        return self._check_templates_for_game_over()


