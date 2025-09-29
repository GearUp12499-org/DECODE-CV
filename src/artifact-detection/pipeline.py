import numpy as np
import cv2
import sys
import math
import os


# CENTER_X = 320
# CENTER_Y = 320

# CONVERSIONS GO HERE
# value from pic125.png
inches2px = lambda inches: inches * 72.85714286
px2inches = lambda px: px / 72.85714286

def getDistance(x, y, r):
    # to impl; we must do some research. Print relative dimensions of a ball with respect to distance
    return 0.0

def canIntake(x, y, r, dist):
    # to impl; given a rect of this size, is the ball in our optimal position?
    # specify distannce theshold, the other data is just for validity.
    # ret boolean
    return True

def detect(img):

    artifacts_found = []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_lower = np.array([84, 230, 80], dtype="uint8")
    green_upper = np.array([87, 255, 215], dtype="uint8")
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    purple_lower = np.array([120, 100, 100], dtype="uint8")
    purple_upper = np.array([150, 255, 255], dtype="uint8")
    mask_purple = cv2.inRange(hsv, purple_lower, purple_upper)

    mask = cv2.bitwise_or(mask_green, mask_purple)

    mask = cv2.GaussianBlur(mask, (9, 9), 2)

    kernel = np.ones((3,3), np.uint8)
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

    output = img.copy()
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            x_offset_in = px2inches(x - center_x)
            y_offset_in = px2inches(y - center_y)

            artifacts_found.append((dist, x_offset_in, y_offset_in, r))

            print(f"Circle: center=({x},{y}), radius={r}px, "f"x_off={x_offset_in:.2f} in, y_off={y_offset_in:.2f} in")

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

            label = f"{x_offset_in:.2f}in, {y_offset_in:.2f}in, r={r}px"

        artifacts_found.sort(key=lambda t: t[0])

    cv2.circle(output, (center_x, center_y), 5, (255, 0, 0), -1)

    return artifacts_found


def runPipeline(img, llrobot):
    try:
        xOff, yOff, w, h, dist, = detect(img)
        intake = canIntake(xOff, yOff, w, h, dist)
            if 

        return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]        

    except Exception as e:
        return np.array([[]]), img, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]