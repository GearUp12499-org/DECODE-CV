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
    [40, 30, 60],
    [98, 245, 255]
]
PURPLE_RANGE = [
    [138, 100, 30],
    [168, 255, 255]
]
THRESHOLD_FDIST = 0.1 # inches threshold for the limelight to signal that intake can start spinning; need to know if this is even needed, or if some intake return type is needed

# Formulas
inches2px = lambda inches: inches * 72.85714286
px2inches = lambda px: px / 72.85714286
fd = lambda r: (-1.45669e9 / ((-290564.256*r) - 40436845.9)) - 16.01634
turn = lambda xOff_in, radius_px: np.degrees(np.arctan2(xOff_in, fd(radius_px)))

# Regression for fd(r); forward distance in inches with respect to pixel radius

# Do not include debug and draw in limelight. Delete all calls
debug_mode = 0
def debug(name, mask):
    if debug_mode == 0:
        cv2.imshow(name, mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return

def draw(img, x, y, r):
    x = int(x)
    y = int(y)
    r = int(r)
    print(f"Drawing at (x = {x}, y = {y}, r = {r})")
    cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    return img

def validMotifRamp(pattern, view):
    breaks = 0
    for i in range(len(view)):
        if not view[i] == pattern[i % 3] and view[i] != 0:
            breaks += 1
    if breaks != 0:
        return False
    return True

def transform(mask):
    ksize = 31
    gse = lambda x, y: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x, y))
    mask = cv2.dilate(mask, gse(3,3), iterations=1)
    mask = cv2.medianBlur(mask, 7)
    debug("Dilation + Median", mask)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 1)
    _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, gse(6,6), iterations=1)
    mask = cv2.medianBlur(mask, 5)

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

def detectRamp(img, pattern):
    artifacts_found = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_lower = np.array(GREEN_RANGE[0], dtype="uint8")
    green_upper = np.array(GREEN_RANGE[1], dtype="uint8")
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    purple_lower = np.array(PURPLE_RANGE[0], dtype="uint8")
    purple_upper = np.array(PURPLE_RANGE[1], dtype="uint8")
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)

    debug("No Gaussian Blur Green", green_mask)
    debug("No Gaussian Blur Purple", purple_mask)
    
    green_mask = transform(green_mask)
    purple_mask = transform(purple_mask)

    debug("Gaussian Blur Green", green_mask)
    debug("Gaussian Blur Purple", purple_mask)

    combined_mask = cv2.bitwise_or(green_mask, purple_mask)
    
    contours = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    circles_list = []
    for contour in contours:
        if len(contour) < 6:
            continue

        circle = find_circle_ransac(contour, max_iterations=1000, inlier_threshold=5.0, min_inliers=10)
        if circle is None:
            continue

        xc, yc, r = circle
        
        if r < 8 or r > 30:
            continue

        mask_region_green = green_mask[max(0, int(yc-r)):min(green_mask.shape[0], int(yc+r)), max(0, int(xc-r)):min(green_mask.shape[1], int(xc+r))]
        mask_region_purple = purple_mask[max(0, int(yc-r)):min(purple_mask.shape[0], int(yc+r)), max(0, int(xc-r)):min(purple_mask.shape[1], int(xc+r))]
        
        green_pixels = np.sum(mask_region_green > 0)
        purple_pixels = np.sum(mask_region_purple > 0)
        
        if green_pixels > purple_pixels:
            color = 2
        elif purple_pixels > green_pixels:
            color = 1
        else:
            color = 0
        
        circles_list.append([xc, yc, r, color])

        cv2.circle(img, (int(xc), int(yc)), int(r), (0, 255, 0), 2)
        cv2.circle(img, (int(xc), int(yc)), 3, (0, 0, 255), -1)

    height, width = img.shape[:2]
    center_x = width // 2
    center_y = height // 2    

    if len(circles_list) > 0:
        circles_list.sort(key=lambda c: c[0], reverse=True)
        
        artifacts_found = []
        ramp = []
        
        for (x, y, r, color) in circles_list:
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
            
            ramp.append(color)
        
        while len(ramp) < 9:
            ramp.append(0)
        
        ramp = ramp[:9]
        
        count = len(circles_list)
        circles = artifacts_found
        valid = validMotifRamp(ramp, pattern)

        return valid, count, ramp, circles

    return None, None, None, None


def runPipeline(img, llrobot):
    try:
        result = detectRamp(img, llrobot)
        
        if result[0] is None:
            return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        valid, count, ramp, circles = result

        if count is not None:
            returnType = 1.0 # acknowledged result
            print(f"ramp: {ramp}, valid: {valid}, count: {count}")
            
            for circle in circles:
                _, xOff_in, yOff_in, radius_in, x_in, y_in, x, y, r = circle
                img = draw(img, x, y, r)
            
            validret = 1.0 if valid else 0.0
            return np.array([[]]), img, [returnType, validret, float(count), 0.0, 0.0, 0.0]

        return np.array([[]]), img, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    except Exception as e:
        print("Error in runPipeline:", e)
        return np.array([[]]), img, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# DO NOT INCLUDE IN LIMELIGHT
if __name__ == "__main__":
    mode = 1
    if mode == 0:
        for i in range(1, 31):
            print(f"=================== image{i}.png ===================")
            img = cv2.imread(f"images5/{i}.png")
            if img is None:
                print(f"Failed to load image{i}.png")
                continue
            llrobot = [1.0, 0.0, 0.0]
            _, img, out = runPipeline(img, llrobot)
            print(f"Output: {out}")
            debug(f"Detection {i}", img)
    else:
        img = cv2.imread(f"images5/ppg.png")
        if img is not None:
            llrobot = [1.0, 1.0, 1.0]
            _, img, out = runPipeline(img, llrobot)
            print(f"Output: {out}")
            debug("Detection", img)
        else:
            print("Failed to load image")