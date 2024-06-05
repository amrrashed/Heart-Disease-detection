import pyautogui
import time

try:
    while True:
        # Get and print the current mouse coordinates
        x, y = pyautogui.position()
        print('Mouse position: x={}, y={}'.format(x, y))
        # Add a small delay to reduce the number of print statements per second
        time.sleep(0.5)
except KeyboardInterrupt:
    print('Exiting...')

