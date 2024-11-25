import time
import pyautogui
import cv2
import numpy as np

class Screen():
    """captures screenshots of the smartphone window
    """
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.new_width = 0
        self.height = 0
        self.new_height = 0
        self.scale = 1

    def find_field(self, image):
        """finds the smartphone window in the screenshot

        Args:
            image ([RGB-image]): screenshot of the entire screen
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([50,100,50])
        upper_green = np.array([80,140,120])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        res = cv2.bitwise_and(hsv, hsv, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        maxsize = 0
        rect = [0,0,0,0]
        for contour in contours:
            (x, y, width, height) = cv2.boundingRect(contour)
            if width*height > maxsize:
                maxsize = width*height
                rect = [x, y, width, height]
        self.x = rect[0]
        self.y = rect[1]
        self.width = rect[2]
        self.height = rect[3]
        self.scale = self.height/750
        self.new_width = int(self.width/self.scale)
        self.new_height = int(self.height/self.scale)

    def get_screenshot(self):
        """Captures a screenshot

        Returns:
            RGB-image: screenshot of the smartphone window
        """
        if self.width == 0:
            image = pyautogui.screenshot()
        else:
            image = pyautogui.screenshot(region=(self.x, self.y, self.width, self.height))
        image = np.array(image)
        if self.width == 0:
            self.find_field(image)
            if self.width != 0:
                image = image[self.y:self.y+self.height, self.x:self.x+self.width]
            else:
                return None
        field = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return field

    def get_image(self):
        """Capture a image of the smartphone window.
        If no window is found, it waits 1s and retries

        Returns:
            RGB-image: resized screenshot of the smartphone window
        """
        while True:
            img_o = self.get_screenshot()
            if img_o is not None:
                break
            else:
                print('no app screen found')
                time.sleep(1)

        img = cv2.resize(img_o, (self.new_width, self.new_height))
        return img
