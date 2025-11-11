import numpy as np
import cv2
import sys
import os

# Code looks a bit smaller; less work is needed due to not accounting for angles. Also may need to improve mask stuff

# Constants
GREEN = 0
PURPLE = 1
BOTH = 2
GREEN_RANGE = [
    [60, 6, 99],
    [90, 245, 255]
]
PURPLE_RANGE = [ 
    [138, 100, 40],
    [168, 255, 255]
]
THRESHOLD_FDIST = 0.1 # inches threshold for the limelight to signal that intake can start spinning
THRESHOLD_ANGLE = 15 # degree magnitude 

# Formulas
inches2px = lambda inches: inches * 72.85714286
px2inches = lambda px: px / 72.85714286
fd = lambda r: 23.38333 * pow(0.989816, r) -3.15906
turn = lambda xOff_in, radius_px: np.degrees(np.arctan2(xOff_in, fd(radius_px)))

# Regression for fd(r); forward distance in inches with respect to pixel radius

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

def transform(mask):
    ksize = 31
    mask = cv2.erode(mask, np.ones((1, 1), np.uint8))
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0) 
    kernel = np.ones((3, 3), np.uint8)
    threshold = 5
    _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    return mask

def find_circle_ransac(points, max_iterations=1000, inlier_threshold=5.0, min_inliers=10):
    best_circle = None
    best_inliers = 0
    points = points.reshape(-1, 2)
    n_points = len(points)

    if n_points < 3:
        return None

    for _ in range(max_iterations):
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]

        A = np.array([p1, p2, p3])
        try:
            m1 = (p1 + p2) / 2
            m2 = (p2 + p3) / 2
            d1 = -(p2[0] - p1[0]) / (p2[1] - p1[1]) if p2[1] != p1[1] else np.inf
            d2 = -(p3[0] - p2[0]) / (p3[1] - p2[1]) if p3[1] != p2[1] else np.inf

            if np.isinf(d1) and np.isinf(d2) or abs(d1 - d2) < 1e-6:
                continue

            if np.isinf(d1):
                xc = m1[0]
                yc = d2 * (xc - m2[0]) + m2[1]
            elif np.isinf(d2):
                xc = m2[0]
                yc = d1 * (xc - m1[0]) + m1[1]
            else:
                xc = (d1 * m1[0] - d2 * m2[0] + m2[1] - m1[1]) / (d1 - d2)
                yc = d1 * (xc - m1[0]) + m1[1]

            r = np.sqrt((p1[0] - xc)**2 + (p1[1] - yc)**2)

            distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
            inliers = np.sum(np.abs(distances - r) < inlier_threshold)

            if inliers > best_inliers and inliers >= min_inliers:
                best_inliers = inliers
                best_circle = (xc, yc, r)

        except Exception:
            continue

    return best_circle

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
    
    mask = transform(mask)
    
    debug("Gaussian Blur", mask)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    circles_list = []
    for contour in contours:
        if len(contour) < 6:
            continue

        circle = find_circle_ransac(contour, max_iterations=1000, inlier_threshold=5.0, min_inliers=10)
        if circle is None:
            continue

        xc, yc, r = circle
        
        if r < 85:
            continue

        circles_list.append([xc, yc, r])

        cv2.circle(img, (int(xc), int(yc)), int(r), (0, 255, 0), 2)
        cv2.circle(img, (int(xc), int(yc)), 3, (0, 0, 255), -1)

    circles = None
    if len(circles_list) > 0:
        circles = np.array([circles_list])

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

            dist_px = np.sqrt(x_offset_px ** 2 + y_offset_px ** 2)
            artifacts_found.append((dist_px, x_offset_in, y_offset_in, radius_in, x_in, y_in, x, y, r))

        artifacts_found.sort(key=lambda t: t[0])
        _, xOff_in, yOff_in, radius_in, x_in, y_in, x, y, r = artifacts_found[0]

        return x_in, y_in, xOff_in, yOff_in, radius_in, x, y, r

    return None, None, None, None, None, None, None, None


def runPipeline(img, llrobot):
    try:
        if llrobot[0] > 0.5:
            x_in, y_in, xOff, yOff, radius, x, y, r = detect(img, GREEN)

            if xOff is not None:
                angle = turn(xOff, r)
                returnType = 1.0 # I see something
                forward = fd(r)
                print("xOff_in:", xOff, "yOff_in:", yOff, "radius_in:", radius, "forward_in:", forward, "angle", angle)
                print("x:", x, "y:", y, "r:", r)
                img = draw(img, x, y, r)
                return np.array([[]]), img, [returnType, xOff, yOff, forward, angle, 0.0, 0.0, 0.0] # SIG --> 2, intakeable; 1, not intakeable

            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # SIG --> 0, nothing

        if llrobot[1] > 0.5:
            x_in, y_in, xOff, yOff, radius, x, y, r = detect(img, PURPLE)

            if xOff is not None:
                angle = turn(xOff, r)
                returnType = 1.0
                forward = fd(r)
                print("xOff_in:", xOff, "yOff_in:", yOff, "radius_in:", radius, "forward_in:", forward, "angle", angle)
                print("x:", x, "y:", y, "r:", r)
                img = draw(img, x, y, r)
                return np.array([[]]), img, [returnType, xOff, yOff, forward, angle, 0.0, 0.0, 0.0] # SIG --> 2, intakeable; 1, not intakeable

            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # SIG --> 0, nothing

        if llrobot[2] > 0.5:
            x_in, y_in, xOff, yOff, radius, x, y, r = detect(img, BOTH)

            if xOff is not None:
                angle = turn(xOff, r)
                returnType = 1.0
                forward = fd(r)
                print("xOff_in:", xOff, "yOff_in:", yOff, "radius_in:", radius, "forward_in:", forward, "angle", angle)
                print("x:", x, "y:", y, "r:", r)
                img = draw(img, x, y, r)
                return np.array([[]]), img, [returnType, xOff, yOff, forward, angle, 0.0, 0.0, 0.0] # SIG --> 2, intakeable; 1, not intakeable

            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # SIG --> 0, nothing

    except Exception as e:
        print("Error in runPipeline:", e)
        return np.array([[]]), img, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # SIG --> -1, error occured

# DO NOT INCLUDE IN LIMELIGHT
if __name__ == "__main__":
    img = cv2.imread("images3/9.png")
    llrobot = [1.0, 0.0, 0.0]
    _, img, _ = runPipeline(img, llrobot)
    debug("Detection", img)