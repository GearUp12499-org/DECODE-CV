import numpy as np
import cv2
import sys
import math
import os

# Code looks a bit smaller; less work is needed due to not accounting for angles. Also may need to improve mask stuff

# Constants
GREEN = 0
PURPLE = 1
BOTH = 2

# Conversions
inches2px = lambda inches: inches * 72.85714286
px2inches = lambda px: px / 72.85714286

# Regression for d(A); forward distance with respect to area; need to calculate this.
fdist = lambda A: A + 1

def canIntake(xOff_in, yOff_in, radius_in):
    area_in2 = math.pi * radius_in * radius_in
    distance_in = math.sqrt(xOff_in ** 2 + fdist(area_in2) ** 2)
    return distance_in < (1.0 + radius_in)

def detect(img, color):
    artifacts_found = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == GREEN:
        green_lower = np.array([84, 230, 80], dtype="uint8")
        green_upper = np.array([87, 255, 215], dtype="uint8")
        mask = cv2.inRange(hsv, green_lower, green_upper)

    if color == PURPLE:
        purple_lower = np.array([120, 100, 100], dtype="uint8")
        purple_upper = np.array([150, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, purple_lower, purple_upper)

    if color == BOTH:
        green_lower = np.array([84, 230, 80], dtype="uint8")
        green_upper = np.array([87, 255, 215], dtype="uint8")
        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        purple_lower = np.array([120, 100, 100], dtype="uint8")
        purple_upper = np.array([150, 255, 255], dtype="uint8")
        mask_purple = cv2.inRange(hsv, purple_lower, purple_upper)

        mask = cv2.bitwise_or(mask_green, mask_purple)

    ksize = 31
    borderType = cv2.BORDER_CONSTANT
    mask = cv2.GaussianBlur(mask, (ksize, ksize), borderType) 
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

    height, width = img.shape[:2]
    center_x = width // 2
    center_y = height // 2

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            x_offset_px = x - center_x
            y_offset_px = y - center_y

            x_offset_in = px2inches(x_offset_px)
            y_offset_in = px2inches(y_offset_px)
            radius_in = px2inches(r) 

            dist_px = math.sqrt(x_offset_px ** 2 + y_offset_px ** 2)
            artifacts_found.append((dist_px, x_offset_in, y_offset_in, radius_in))

        artifacts_found.sort(key=lambda t: t[0]) # maybe change for later(?)
        _, xOff_in, yOff_in, radius_in = artifacts_found[0]

        area_in2 = math.pi * radius_in * radius_in
        distance_in = math.sqrt(xOff_in ** 2 + fdist(area_in2) ** 2)
        turn_angle = math.atan(xOff_in/distance_in) # how much turn is needed to face the artifact, radians

        return xOff_in, yOff_in, radius_in, turn_angle

    return None, None, None, None


def runPipeline(img, llrobot):
    try:
        if llrobot[0] > 0.5:
            xOff, yOff, radius, turn_angle = detect(img, GREEN)

            if xOff is not None:
                intakeable = canIntake(xOff, yOff, radius)
                returnType = 2.0 if intakeable else 1.0
                return np.array([[]]), img, [returnType, xOff, yOff, turn_angle, 0.0, 0.0, 0.0, 0.0]

            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if llrobot[1] > 0.5:
            xOff, yOff, radius, turn_angle = detect(img, PURPLE)

            if xOff is not None:
                intakeable = canIntake(xOff, yOff, radius)
                returnType = 2.0 if intakeable else 1.0
                return np.array([[]]), img, [returnType, xOff, yOff, turn_angle, 0.0, 0.0, 0.0, 0.0]

            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if llrobot[2] > 0.5:
            xOff, yOff, radius, turn_angle = detect(img, BOTH)

            if xOff is not None:
                intakeable = canIntake(xOff, yOff, radius)
                returnType = 2.0 if intakeable else 1.0
                return np.array([[]]), img, [returnType, xOff, yOff, turn_angle, 0.0, 0.0, 0.0, 0.0]

            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    except Exception as e:
        print("Error in runPipeline:", e)
        return np.array([[]]), img, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# DO NOT INCLUDE IN LIMELIGHT
if __name__ == "__main__":
    img = cv2.imread("images/snap027998558882.png")
    llrobot = [0.0, 0.0, 1.0]
    runPipeline(img, llrobot)