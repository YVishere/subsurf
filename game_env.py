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

    def __init__(self, frame_stack=4, frame_size=(84, 84), frame_skip=2):
        super(SubwayEnv, self).__init__()
        global ss

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
            'game_over': -10,
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
        self._restart_game()
        
        self.score = 0
        self.previous_score = -10
        self.steps = 0
        self.game_over = False
        self.prev_score_region = None  # Reset detection state
        self.score_change_counter = 0  # Reset counter
        
        self.frames = np.zeros(self.obs_shape, dtype=np.uint8)
        
        # Get initial observation
        for i in range(self.frame_stack):
            self.frames[i], self.non_resized = self._get_observation()
        
        # Give game time to fully load - longer wait
        time.sleep(2.0)  # Increased from 1.5
        
        # Try clicking play button again after initial wait
        start_game()
        time.sleep(0.5)
        
        # Get fresh frames after second click
        for i in range(self.frame_stack):
            self.frames[i], self.non_resized = self._get_observation()
    
        # Verify game started properly
        self.score = self._extract_score()
        is_game_over = False  # Don't check game over during reset
    
        restart_attempts = 0
        while self.score < 0 and restart_attempts < 5:  # Only check score, not game over
            print(f"Game may not have restarted properly. Trying again... (Attempt {restart_attempts+1}/5)")
            restart_attempts += 1
            
            # Try clicking the restart/play button with increased delay
            start_game()
            time.sleep(self.reset_delay)
            
            # Check again
            for i in range(self.frame_stack):
                self.frames[i], self.non_resized = self._get_observation()
                
            self.score = self._extract_score()
    
        # If still not working after attempts, continue anyway
        if self.score < 0:
            print("Warning: Score still negative, but continuing...")
            self.score = 0  # Force valid score
    
        return self.frames, {}
    
    def step(self, action):
        reward_sum = 0
        done = False
        
        # Skip frames to speed up gameplay
        for _ in range(self.frame_skip):
            # Get fresh observation
            new_frame, self.non_resized = self._get_observation()
            
            # Check game over with fresh frame
            self.game_over, self.case = self._detect_game_over()
            
            # Only execute action on first frame of skip sequence and if game not over
            if _ == 0 and not self.game_over:
                self.actions[action]()
            
            # Update frame stack with latest frame
            self.frames = np.roll(self.frames, -1, axis=0)
            self.frames[-1] = new_frame
            
            # Calculate immediate reward
            immediate_reward = self._calculate_reward() if not self.game_over else -10
            reward_sum += immediate_reward
            
            # Break out of frame skip if game ends
            if self.game_over:
                done = True
                break
                
            # Small wait between skipped frames
            time.sleep(0.01)
        
        self.steps += 1
        
        # Score updates (less frequent)
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
        """Multi-method game over detection for better reliability"""
        # Try template matching first (most reliable)
        if self.score > 0:
            return False, 0

        if self._check_templates_for_game_over():
            return True, 0
    
        # Then check score region for changes (your current approach)
        global xST, xED, yST, yED
        
        # Extract score region with more padding
        padding = 15  # Increased padding
        y_start = max(0, yST-padding)
        y_end = min(self.non_resized.shape[0], yED+padding*2)  # More padding below
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
        
        # Ensure regions are the same size
        if gray_region.shape != self.prev_score_region.shape:
            self.prev_score_region = cv2.resize(self.prev_score_region, 
                                              (gray_region.shape[1], gray_region.shape[0]))
        
        # Calculate difference with more robust method
        diff = cv2.absdiff(gray_region, self.prev_score_region)
        mean_diff = np.mean(diff)
        
        # Also check for big changes in specific UI areas (more reliable)
        # Update reference with blend for stability
        self.prev_score_region = cv2.addWeighted(self.prev_score_region, 0.6, gray_region, 0.4, 0)
        
        # Use a more reliable threshold for game over
        if mean_diff > self.score_diff_threshold * 1.5:  # Much higher threshold for confidence
            print(f"Game over detected: Large UI change {mean_diff:.2f}")
            return True, 0
            
        # Fall back to score checks only as last resort
        if self.score < 0 or (self.score == 7 and self.previous_score == 7):
            return True, 0
        
        return False, 2
    
    def _check_templates_for_game_over(self):
        """Check for game over screen using template matching"""
        global template, template2, template3, template4
        
        # Get smaller version of screenshot for faster processing
        screenshot = cv2.cvtColor(self.non_resized, cv2.COLOR_BGR2GRAY)
        small_screen = cv2.resize(screenshot, (0, 0), fx=0.3, fy=0.3)
        
        templates = [template, template2, template3, template4]
        for tmpl in templates:
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
                    if max_val > 0.7:  # Increased from 0.6
                        print(f"Game over detected by template matching: {max_val:.2f}")
                        return True
            except Exception as e:
                continue
        
        return False
    
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
        """Improved restart with better timing"""
        restart_game(self.case)
        
        # Wait a moment for restart animation
        time.sleep(1.0)
        
        # Ensure window is active
        window = get_bluestacks_window()
        if window:
            # Additional click to ensure game starts
            x, y = to_absolute_coords(window, option=1)
            pyautogui.click(x, y)
            
        time.sleep(0.5)


