import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('assets/test_image.jpg',)

"""

Step 1: Convert images to gray scale so that pixels will range between 0 to 255, and also less computational power required

Edge Detection
- Identifying sharp changes in intensity in adjacent pixels

Gradient
- Measure of change in brightness over adjacent pixels

Edge
- rapid changes in brightness

"""

lane_image = np.copy(image)
gray = cv.cvtColor(lane_image, cv.COLOR_RGB2GRAY)


"""

Step 2: Reduce Noise

Gaussian Blur
- to smoothen images and reduce noise

"""

blur = cv.GaussianBlur(gray, (5,5), 0)


"""
Step 3: Finding Lane Lines using Canny

- by default, canny applies a 5x5 kernel of gaussian blur
- measures adjacent changes in intesnsity in all directions (x and y)
- large change, large derivative
- it traces the edge with large change in intensity (large gradient) in an outline of white pixels.

"""

canny = cv.Canny(blur, 50, 150)


"""
Step 4: Region of Interest (Finding Lane Lines part 1)

- use matplotlib to analyze the shape of the ROI
- subsequently get the coordinates required to create a mask


Step 5: Bitwise_and (Finding Lane Lines part 2)

Step 6: Hough Transform (Finding Lane Lines part 3)

"""

# plt.imshow(canny_image)
# plt.show()

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] 
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0: 
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0) # to get average slope and intercept, axis=0 goes vertically
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])


def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    canny = cv.Canny(blur, 50, 150)

    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines: 
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
          [(200, height), (1100, height), (550, 250)]
        ])
    mask = np.zeros_like(image) # create an array of zeros with same shape as image
    cv.fillPoly(mask, polygons, 255)
    masked_image = cv.bitwise_and(image, mask)

    return masked_image



# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# overlay_original = cv.addWeighted(lane_image, 0.8, line_image, 1, 1)


# cv.imshow('canny image', overlay_original)
# cv.waitKey(0)


cap = cv.VideoCapture("assets/test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    overlay_original = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    cv.imshow('canny image', overlay_original)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()