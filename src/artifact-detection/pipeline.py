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
GREEN_RANGE = [[60, 6, 99], [69, 245, 255]]
PURPLE_RANGE = [[120, 100, 100], [150, 255, 255]]


# Conversions
inches2px = lambda inches: inches * 72.85714286
px2inches = lambda px: px / 72.85714286

# Regression for d(A); forward distance with respect to area; need to calculate this.
# Need to compare area in in2 with distance in inches
fdist = lambda A: A + 1

# Do not include debug and draw in limelight. Delete all calls
def debug(name, mask):
    cv2.imshow(name, mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw(img, x, y, r):
    x = int(x)
    y = int(y)
    r = int(r)
    print(f"Drawing at x={x}, y={y}, r={r}")
    cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    return img

def canIntake(xOff_in, yOff_in, radius_in):
    area_in2 = math.pi * radius_in * radius_in
    distance_in = math.sqrt(xOff_in ** 2 + fdist(area_in2) ** 2)
    return distance_in < (1.0 + radius_in)

def detect(img, color):
    artifacts_found = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == GREEN:
        green_lower = np.array(GREEN_RANGE[0], dtype="uint8")
        green_upper = np.array(GREEN_RANGE[1], dtype="uint8")
        mask = cv2.inRange(hsv, green_lower, green_upper)

    if color == PURPLE:
        purple_lower = np.array(PURPLE_RANGE[0], dtype="uint8")
        purple_upper = np.array(PURPLE_RANGE[1], dtype="uint8")
        mask = cv2.inRange(hsv, purple_lower, purple_upper)

    if color == BOTH:
        green_lower = np.array(GREEN_RANGE[0], dtype="uint8")
        green_upper = np.array(GREEN_RANGE[1], dtype="uint8")
        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        purple_lower = np.array(PURPLE_RANGE[0], dtype="uint8")
        purple_upper = np.array(PURPLE_RANGE[1], dtype="uint8")
        mask_purple = cv2.inRange(hsv, purple_lower, purple_upper)

        mask = cv2.bitwise_or(mask_green, mask_purple)
        
    debug("No Gaussian Blur", mask)
    
    # may need to do more here due to masking
    ksize = 31
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0) 
    kernel = np.ones((3, 3), np.uint8)
    threshold = 5
    _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    debug("Gaussian Blur", mask)

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
            img = draw(img, x, y, r)

            x_in = px2inches(x)
            y_in = px2inches(y)

            x_offset_px = x - center_x
            y_offset_px = y - center_y

            x_offset_in = px2inches(x_offset_px)
            y_offset_in = px2inches(y_offset_px)
            radius_in = px2inches(r) 

            dist_px = math.sqrt(x_offset_px ** 2 + y_offset_px ** 2)
            artifacts_found.append((dist_px, x_offset_in, y_offset_in, radius_in, x_in, y_in, x, y, r))

        artifacts_found.sort(key=lambda t: t[0]) # maybe change for later (?)
        _, xOff_in, yOff_in, radius_in, x_in, y_in, x, y, r = artifacts_found[0]

        area_in2 = math.pi * radius_in * radius_in
        distance_in = math.sqrt(xOff_in ** 2 + fdist(area_in2) ** 2)
        turn_angle = math.atan(xOff_in/distance_in) # how much turn is needed to face the artifact, radians

        return x_in, y_in, xOff_in, yOff_in, radius_in, turn_angle, x, y, r

    return None, None, None, None, None, None, None, None, None


def runPipeline(img, llrobot):
    try:
        if llrobot[0] > 0.5:
            x_in, y_in, xOff, yOff, radius, turn_angle, x, y, r = detect(img, GREEN)

            if xOff is not None:
                intakeable = canIntake(xOff, yOff, radius)
                returnType = 2.0 if intakeable else 1.0
                print("xOff_in:", xOff, "yOff_in:", yOff, "radius_in:", radius)
                print("x:", x, "y:", y, "r:", r)
                img = draw(img, x, y, r)
                return np.array([[]]), img, [returnType, xOff, yOff, turn_angle, 0.0, 0.0, 0.0, 0.0]

            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if llrobot[1] > 0.5:
            x_in, y_in, xOff, yOff, radius, turn_angle, x, y, r = detect(img, PURPLE)

            if xOff is not None:
                intakeable = canIntake(xOff, yOff, radius)
                returnType = 2.0 if intakeable else 1.0
                print("xOff_in:", xOff, "yOff_in:", yOff, "radius_in:", radius)
                print("x:", x, "y:", y, "r:", r)
                img = draw(img, x, y, r)
                return np.array([[]]), img, [returnType, xOff, yOff, turn_angle, 0.0, 0.0, 0.0, 0.0]

            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if llrobot[2] > 0.5:
            x_in, y_in, xOff, yOff, radius, turn_angle, x, y, r= detect(img, BOTH)

            if xOff is not None:
                intakeable = canIntake(xOff, yOff, radius)
                returnType = 2.0 if intakeable else 1.0
                print("xOff_in:", xOff, "yOff_in:", yOff, "radius_in:", radius)
                print("x:", x, "y:", y, "r:", r)
                img = draw(img, x, y, r)
                return np.array([[]]), img, [returnType, xOff, yOff, turn_angle, 0.0, 0.0, 0.0, 0.0]

            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    except Exception as e:
        print("Error in runPipeline:", e)
        return np.array([[]]), img, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# DO NOT INCLUDE IN LIMELIGHT
if __name__ == "__main__":
    img = cv2.imread("images2/8.png")
    llrobot = [1.0, 0.0, 0.0]
    _, img, _ = runPipeline(img, llrobot)
    debug("Detection", img)