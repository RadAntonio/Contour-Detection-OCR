import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
import pytesseract

def show_img(img):
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def find_contours(img):
    conts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key = cv2.contourArea, reverse = True)[:6]
    return conts

def sort_points(points):
    points = points.reshape((4,2))
    new_points = np.zeros((4, 1, 2), dtype = np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]
    dif = np.diff(points, axis = 1)
    new_points[1] = points[np.argmin(dif)]
    new_points[3] = points[np.argmax(dif)]
    return new_points

img = cv2.imread('./images/test1.jpg')
original = img.copy()
show_img(img)
(H, W) = img.shape[:2]
print(H, W)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_img(gray)

blur = cv2.GaussianBlur(gray, (5, 5), 0)
show_img(blur)

edged = cv2.Canny(blur, 60, 160)
show_img(edged)

conts = find_contours(edged.copy())
print(conts)

for c in conts:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approximation) == 4:
        larger = approximation
        break

print(larger)
cv2.drawContours(img, larger, -1, (120, 255, 0), 28)
cv2.drawContours(img, [larger], -1, (120, 255, 0), 2)
show_img(img)

points_larger = sort_points(larger)
print(points_larger)

pts1 = np.float32(points_larger)
pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(matrix)

transform = cv2.warpPerspective(original, matrix, (W, H))
show_img(transform)

config_tesseract = "--tessdata-dir tessdata"

increase = cv2.resize(transform, None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
show_img(increase)

brightness = 50
contrast = 80
adjust = np.int16(transform)

adjust = adjust * (contrast / 127 + 1) - contrast + brightness
adjust = np.clip(adjust, 0, 255)
adjust = np.uint8(adjust)
show_img(adjust)

processed_img = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
show_img(processed_img)

margin = 18
img_edges = processed_img[margin: H - margin, margin: W - margin]
show_img(img_edges)

text = pytesseract.image_to_string(increase, lang='eng', config='--tessdata-dir tessdata')
print(text)

