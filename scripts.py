import pygetwindow as gw
import pyautogui
import time
from typing import Optional, Dict, Tuple
import ctypes

playbutton = (0.716, 0.941)
continuebutton = (0.434, 0.701)

# Import Windows API constants and functions
SendInput = ctypes.windll.user32.SendInput
# Define constants for keyboard events
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002

# Define key codes
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_UP = 0x26
VK_DOWN = 0x28

# Create input structure
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def press_key(key_code):
    """Send a key press using Windows API directly"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(key_code, 0, KEYEVENTF_KEYDOWN, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(key_code):
    """Send a key release using Windows API directly"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(key_code, 0, KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def fast_key_press(key_code):
    """Press and release a key quickly"""
    press_key(key_code)
    time.sleep(0.01)
    release_key(key_code)
    return True

def get_bluestacks_window() -> Optional[Dict]:

    bluestacks_windows = gw.getWindowsWithTitle('BlueStacks')
    
    if not bluestacks_windows:
        print("Error: BlueStacks window not found")
        return None
    
    # Get the first matching window
    bluestacks_window = bluestacks_windows[0]
    
    # Try to activate the window (optional)
    try:
        bluestacks_window.activate()
    except Exception as e:
        # Often this still succeeds despite error
        if "Error code from Windows: 0" in str(e):
            print("Window activation reported an error but seems to have succeeded")
        else:
            print(f"Warning: Window activation failed: {e}")
    
    return {
        'left': bluestacks_window.left,
        'top': bluestacks_window.top,
        'width': bluestacks_window.width,
        'height': bluestacks_window.height,
        'window': bluestacks_window
    }

def to_absolute_coords(window_info: Dict, option = 0, relx = 0, rely = 0) -> Tuple[int, int]:
    if option == 0:
        global playbutton
        rel_x, rel_y = playbutton
    elif option == 1:
        global continuebutton
        rel_x, rel_y = continuebutton
    else:
        rel_x = relx
        rel_y = rely

    abs_x = window_info['left'] + int(rel_x * window_info['width'])
    abs_y = window_info['top'] + int(rel_y * window_info['height'])
    return abs_x, abs_y


def restart_game(case, count = 1):
    #Press continue button first
    info = get_bluestacks_window()

    if info is None:
        print("Error: BlueStacks window not found")
        return
    
    # if case == 0:
    #     abs_x, abs_y = to_absolute_coords(info, 1)
    #     pyautogui.moveTo(abs_x, abs_y, duration=0.05) 
    #     pyautogui.click() 

    #     time.sleep(0.1)  # Short delay to ensure the click is registered

    #do it twice
    # Press play button

    # time.sleep(2)
    # pyautogui.press('space')
    # time.sleep(3)

    while count > 0:
        abs_x, abs_y = to_absolute_coords(info, 0)
        pyautogui.moveTo(abs_x, abs_y, duration=0.05)
        time.sleep(0.1)  # Short delay to ensure the click is registered
        pyautogui.click()

        time.sleep(0.1)  # Short delay to ensure the click is registered
        center_x = info['left'] + info['width']//2
        center_y = info['top'] + info['height']//2
        pyautogui.moveTo(center_x, center_y, duration=0.05)
        count -= 1

        if count > 0:
            time.sleep(2)  # Wait for the game to load 

def start_game():
    # Press play button
    info = get_bluestacks_window()

    if info is None:
        print("Error: BlueStacks window not found")
        return
    
    abs_x, abs_y = to_absolute_coords(info, 0)
    pyautogui.moveTo(abs_x, abs_y, duration=0.05)
    pyautogui.click()
    time.sleep(0.1)  # Short delay to ensure the click is registered
    center_x = info['left'] + info['width']//2
    center_y = info['top'] + info['height']//2
    pyautogui.moveTo(center_x, center_y, duration=0.05)

def swipe_bluestacks(start_x, start_y, end_x, end_y, duration=0.08):
    """Performs a faster swipe from start to end coordinates"""
    try:
        # Move to start position without clicking (reduced timing)
        pyautogui.moveTo(start_x, start_y, duration=0.02)
        
        # Reduced pause
        time.sleep(0.02)
        
        # Perform the swipe
        pyautogui.mouseDown(button='left')
        
        # Reduced steps for faster execution
        steps = max(5, int(duration * 40))  # Reduced from 10 steps to 5 minimum
        
        for i in range(1, steps + 1):
            progress = i / steps
            # Smaller curve for faster movement
            curve_height = 3  # Reduced from 5
            curve_offset = curve_height * progress * (1 - progress) * 4
            
            # Calculate position along path with curve
            if progress < 0.5:
                actual_progress = 2 * progress * progress
            else:
                actual_progress = 1 - (2 * (1 - progress) * (1 - progress))
            
            next_x = start_x + (end_x - start_x) * actual_progress
            next_y = start_y + (end_y - start_y) * actual_progress - curve_offset
            
            pyautogui.moveTo(next_x, next_y)
            
            # Reduced sleep time
            time.sleep(duration / (steps * 1.5))
        
        # Quick finish
        pyautogui.moveTo(end_x, end_y, duration=0.01)
        pyautogui.mouseUp(button='left')
        
        # Minimal pause after completion
        time.sleep(0.02)
        return True
        
    except Exception as e:
        print(f"Swipe failed: {e}")
        try:
            pyautogui.mouseUp(button='left')
        except:
            pass
        return False

def swipe_left_bluestacks(distance=70):
    print("Left")
    return fast_key_press(VK_LEFT)

def swipe_right_bluestacks(distance=70):
    print("Right")
    return fast_key_press(VK_RIGHT)

def swipe_up_bluestacks(distance=70):
    print("Up")
    return fast_key_press(VK_UP)

def swipe_down_bluestacks(distance=70):
    print("Down")
    return fast_key_press(VK_DOWN)

def random_ac(i):
    match i:
        case 1:
            swipe_left_bluestacks(70)
        case 2:
            swipe_right_bluestacks(70)
        case 3:
            swipe_up_bluestacks(70)
        case 4:
            swipe_down_bluestacks(70)

# Execute the swipe
if __name__ == "__main__":
    import random 
    count =0 
    time.sleep(2)
    while count < 10:
        count += 1
        i = random.randint(1, 4)
        random_ac(i)
        print(i)
    
    # swipe_left_bluestacks(70)
    # Uncomment to test other swipes
    # swipe_right_bluestacks(70)
    # swipe_up_bluestacks(70)
    # swipe_down_bluestacks(70)
