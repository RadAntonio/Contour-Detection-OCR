import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import os
from easyocr import Reader

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect



output_dir = './saved_images'

img_var = 'test18'
#thresh_val = 80

image1 = cv2.imread(rf'./images/{img_var}.jpg')
img_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
#ret, thresh1 = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY) # for manual threshold value adjustment
otsu_threshold, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Original Image')


plt.subplot(2, 3, 2)
plt.imshow(img_gray, cmap='gray')
plt.title('Gray Image')


plt.subplot(2, 3, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Threshold Image')

contours1, hierarchy1 = cv2.findContours(image=binary_image,
                                         mode=cv2.RETR_TREE,
                                         method=cv2.CHAIN_APPROX_NONE)

image_contours1 = cv2.drawContours(image=image1.copy(),
                                   contours=contours1,
                                   contourIdx=-1,
                                   color=(255, 255, 0),
                                   thickness=2,
                                   lineType=cv2.LINE_AA)

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(image_contours1, cv2.COLOR_BGR2RGB))
plt.title('Contour Image')

largest_contour = max(contours1, key=cv2.contourArea)
image_with_largest_contour = cv2.drawContours(image1.copy(), [largest_contour], -1, (0, 255, 0), 3)

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(image_with_largest_contour, cv2.COLOR_BGR2RGB))
plt.title('Largest Contour')

epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

if len(approx) == 4:
    points = approx.reshape(4, 2)
    ordered_points = order_points(points)

    (tl, tr, br, bl) = ordered_points
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    max_height = max(int(height_left), int(height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_points, dst)
    warped = cv2.warpPerspective(image1, M, (max_width, max_height))

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title('Warped Image')
else:
    print("Could not find a quadrilateral contour for the receipt.")

#plt.suptitle(f'thresh val = {thresh_val}')
plt.tight_layout()
final_path = os.path.join(output_dir, f'{img_var}_final_plot.png')
plt.savefig(final_path)
plt.show()

config_tesseract = "--tessdata-dir ./tessdata"
result = pytesseract.image_to_string(warped, lang='eng', config=config_tesseract)
print(result)

print('--------------------------------------------------------------')

languages_list = ['en']
reader = Reader(languages_list)
results = reader.readtext(warped)
for r in results:
    print(r[1])



