import cv2
import numpy as np
import math

import cv2
import numpy as np
import math

inches2px = lambda inches: inches * 72.85714286

def detect_and_visualize(img, in_per_px=0.01, show=True):

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

            x_offset_in = inches2px(x - center_x)
            y_offset_in = inches2px(y - center_y)

            artifacts_found.append((dist, x_offset_in, y_offset_in, r))

            print(f"Circle: center=({x},{y}), radius={r}px, "f"x_off={x_offset_in:.2f} in, y_off={y_offset_in:.2f} in")

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

            label = f"{x_offset_in:.2f}in, {y_offset_in:.2f}in, r={r}px"

        artifacts_found.sort(key=lambda t: t[0])

    cv2.circle(output, (center_x, center_y), 5, (255, 0, 0), -1)

    if show:
        cv2.imshow("Detected Circles", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return artifacts_found

if __name__ == "__main__":
    img = cv2.imread("images/snap215019014291.png")
    print(detect_and_visualize(img))