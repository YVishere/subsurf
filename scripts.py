import pygetwindow as gw
import pyautogui
import time
from typing import Optional, Dict, Tuple

playbutton = (0.716, 0.941)
continuebutton = (0.434, 0.701)

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


def restart_game(case):
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

    # Press play button
    abs_x, abs_y = to_absolute_coords(info, 0)
    pyautogui.moveTo(abs_x, abs_y, duration=0.05)
    pyautogui.click()

    time.sleep(0.1)  # Short delay to ensure the click is registered
    center_x = info['left'] + info['width']//2
    center_y = info['top'] + info['height']//2
    pyautogui.moveTo(center_x, center_y, duration=0.05)

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
    pyautogui.click()
    pyautogui.press('left')
    print("Down arrow key pressed")
    result = True
    return result

def swipe_right_bluestacks(distance=70):
    pyautogui.click()
    pyautogui.press('right')
    print("Down arrow key pressed")
    result = True
    return result

def swipe_up_bluestacks(distance=70):
    pyautogui.click()
    pyautogui.press('up')
    print("Down arrow key pressed")
    result = True
    return result

def swipe_down_bluestacks(distance=70):
    pyautogui.click()
    pyautogui.press('down')

    print("Down arrow key pressed")
    result = True
    return result

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
