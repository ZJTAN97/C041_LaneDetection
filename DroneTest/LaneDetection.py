import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


image = cv.imread('DroneImages/test3.jpg')
image = cv.resize(image, (500,500))
image_copy = np.copy(image)

def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    canny = cv.Canny(blur, 50, 150)

    return canny


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] 
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines: 
            cv.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 10)

    return line_image


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



canny_image = canny(image_copy)
lines = cv.HoughLinesP(canny_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(image_copy, lines)
line_image = display_lines(image_copy, averaged_lines)
overlay = cv.addWeighted(image_copy, 0.8, line_image, 1, 1)



# video = cv.VideoCapture('DroneImages/video_test1.mp4')
# i = 0
# while(video.isOpened()):
#     _, frame = video.read()
#     canny_image = canny(frame)
#     lines = cv.HoughLinesP(canny_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines = average_slope_intercept(frame, lines)
#     line_image = display_lines(frame, averaged_lines)
#     overlay = cv.addWeighted(frame, 0.8, line_image, 1, 1)

#     cv.imwrite(f'./drone_train/test{i+1}.jpg', frame)
#     cv.imwrite(f'./drone_mask/test{i+1}.jpg', line_image)

#     i += 1


#     cv.imshow('canny image', overlay)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# video.release()
# cv.destroyAllWindows()



# cv.imwrite('test7.jpg', line_image)

cv.imshow('Overlay Image', overlay)
cv.imshow('Line Image', line_image)
cv.waitKey(0)