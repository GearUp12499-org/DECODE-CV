import cv2
import numpy as np
import math

inches2px = lambda inches: inches * 72.85714286
px2inches = lambda px: px / 72.85714286

def detect_circles(img):
    artifacts_found = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_lower = np.array([85, 205, 80], dtype="uint8")
    green_upper = np.array([87, 255, 215], dtype="uint8")
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    purple_lower = np.array([120, 100, 100], dtype="uint8")
    purple_upper = np.array([150, 255, 255], dtype="uint8")
    mask_purple = cv2.inRange(hsv, purple_lower, purple_upper)

    mask = cv2.bitwise_or(mask_green, mask_purple)

    mask = cv2.GaussianBlur(mask, (9, 9), 2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=20,
        minRadius=10,
        maxRadius=0
    )

    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        x, y, r = circles[0]
        
        x_offset_px = x - center_x
        y_offset_px = y - center_y
        
        x_offset_in = px2inches(x_offset_px)
        y_offset_in = px2inches(y_offset_px)
        
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        
        return x_offset_in, y_offset_in, r
    
    return None, None, None

def runPipeline(img, llrobot):
    try:
        center_x = img.shape[1] // 2
        center_y = img.shape[0] // 2
        cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
        
        xOff, yOff, radius = detect_circles(img)
        
        if xOff is not None:
            return np.array([[]]), img, [1.0, xOff, yOff, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    except Exception as e:
        return np.array([[]]), img, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]