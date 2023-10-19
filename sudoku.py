import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.5):
    # For sudoku image
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)




# Read the image
img = cv.imread('sudoku.jpg')
img_width = int(img.shape[0])
img_height = int(img.shape[1])

# Turn image to grayscale, and resize if needed
if (img_width > 1000 and img_height > 1000):
    img = rescaleFrame(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
else:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Use Gaussian blur and thresholding to return sharper edges
blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_CONSTANT)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 11)


# Invert the colour of the image
inverted_image = cv.bitwise_not(adaptive_thresh)
cv.imshow('Adaptive Thresholding', inverted_image)

# Find contours in order to get the corners of the grid (largest area most likely to be corner of sudoku grid)
contours, hierarchies = cv.findContours(inverted_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
filtered_contours = []
for cnt in contours:
    if cv.contourArea(cnt) > 100:
        filtered_contours.append(cnt)

filtered_contours = sorted(filtered_contours, key=cv.contourArea, reverse=True)

# Approximate the rectangle
corners = filtered_contours[:4]
epsilon = 0.02 * cv.arcLength(corners[0], True)
approx = cv.approxPolyDP(corners[0], epsilon, True)

print(approx)

# Draw the rectangle around the original image
cv.drawContours(img, [approx], -1, (0, 255, 0), 2)

# Display the image
cv.imshow('Sudoku Grid', img)

# Crop the image to the approx vertices
pts1 = np.float32([approx[0], approx[1], approx[2], approx[3]])
pts2 = np.float32([[0, 0], [0, 500], [500, 500], [500, 0]])
matrix = cv.getPerspectiveTransform(pts1, pts2)
cropped_img = cv.warpPerspective(img, matrix, (500, 500))

# Display the cropped image
cv.imshow('Cropped Image', cropped_img)

print(corners)




cv.waitKey(0)