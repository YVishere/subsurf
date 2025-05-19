from google import genai
import pygetwindow as gw
import pyautogui
from PIL import Image
import time
import base64
import ollama


API = "AIzaSyCaAEMD3UyCQEPwEaRL6cQwR3bc5kwgWzk"
model = genai.Client(api_key=API)


def get_info(screenshot):
    """
    Get information about the game from a screenshot.
    :param screenshot: The screenshot of the game.
    :return: The information about the game.
    """
    # Use the Gemini model to analyze the screenshot
        
    resp = model.models.generate_content(
        model="gemini-2.0-flash",
        contents=["In context of Subway Surfers gameplay. Classify this image as and only output: Game over, Game on, neither", screenshot],
    ).text
    return resp

def take_screenshot():
    """
    Take a screenshot of the game.
    :return: The screenshot of the game.
    """
    # Use the Gemini model to take a screenshot
    bluestacks_windows = gw.getWindowsWithTitle('BlueStacks')

    if not bluestacks_windows:
        print("BlueStacks window not found")
        return None
    else:
        # Get the first window that matches
        bluestacks_window = bluestacks_windows[0]
        
        # Try to activate the window, but handle the potential error
        try:
            bluestacks_window.activate()
        except Exception as e:
            # If the error message indicates it was actually successful, continue
            if "Error code from Windows: 0" in str(e):
                print("Window activation reported an error but seems to have succeeded")
            else:
                print(f"Window activation failed: {e}")
        
        # Get the position and size of the window
        left, top, width, height = bluestacks_window.left, bluestacks_window.top, bluestacks_window.width, bluestacks_window.height
        
        # Take a screenshot of the specified region
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        
        # Now 'screenshot' is a PIL Image variable you can use
        # You can save it if needed
        screenshot.save('bluestacks_screenshot.png')
        
        print(f"Screenshot taken. Size: {width}x{height}")
    return screenshot

def main():
    # Take a screenshot of the game
    screenshot = take_screenshot()
    
    # Get information about the game from the screenshot
    if screenshot is None:
        print("Failed to take screenshot.")
        exit(1)
    
        
    info = get_info(screenshot)
    # info = local()
    
    # Print the information about the game
    print(info)

    toRet = [0,0,0]
    if "Game over" in info:
        toRet[0] = 1
    elif "Game on" in info:
        toRet[1] = 1
    else:
        toRet[2] = 1

    

def local():
    paths = ["./bluestacks_screenshot.png"]
    for img_path in paths:
        print(f"Processing image: {img_path}")
        try:
            # Read the image file as binary data
            with open(img_path, 'rb') as img_file:
                img_data = img_file.read()
            
            # Convert image to base64 for Ollama
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Use the correct approach as per Gemma 3 documentation - images at top level
            response = ollama.generate(
                model='gemma3:12b',
                prompt="Cutput only Game on, Game over, or neither",
                images=[img_base64],  # Pass base64 encoded image data at top level
                options={"temperature": 0.1},  # Lower temperature for more consistent output
            )
            
            # Extract the caption from the response
            caption = response['response'].strip()
            return caption
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

if __name__ == "__main__":
    start = time.time()
    while True:
        main()
    end = time.time()
    print(f"Execution time: {end - start} seconds")