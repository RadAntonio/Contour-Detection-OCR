import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import os
from easyocr import Reader
from paddleocr import PaddleOCR


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def preprocess_receipt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bw_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    return bw_image

output_dir = './saved_images'

img_var = 'test31'
#thresh_val = 80

image1 = cv2.imread(rf'./images/{img_var}.jpg')
img_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
#ret, thresh1 = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY) # for manual threshold value adjustment
otsu_threshold, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



plt.figure(figsize=(20, 10))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Original Image')


plt.subplot(2, 4, 2)
plt.imshow(img_gray, cmap='gray')
plt.title('Gray Image')


plt.subplot(2, 4, 3)
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

plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(image_contours1, cv2.COLOR_BGR2RGB))
plt.title('Contour Image')

largest_contour = max(contours1, key=cv2.contourArea)
image_with_largest_contour = cv2.drawContours(image1.copy(), [largest_contour], -1, (0, 255, 0), 3)

plt.subplot(2, 4, 5)
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

    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title('Warped Image')
    warped_bw = preprocess_receipt(warped)
    plt.subplot(2, 4, 7)
    plt.imshow(cv2.cvtColor(warped_bw, cv2.COLOR_BGR2RGB))
    plt.title('Warped Image Black & White')
    # opening = cv2.dilate(warped2, np.ones((2, 2), np.uint8))
    # plt.subplot(2, 4, 7)
    # plt.imshow(cv2.cvtColor(opening, cv2.COLOR_BGR2RGB))
    # plt.title('Erosion Image')
    erosion = cv2.erode(warped_bw, np.ones((2,2), np.uint8))
    plt.subplot(2, 4, 8)
    plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
    plt.title('Erosion Image')

else:
    print("Could not find a quadrilateral contour for the receipt.")

#plt.suptitle(f'thresh val = {thresh_val}')
plt.tight_layout()
final_path = os.path.join(output_dir, f'{img_var}_final_plot.png')
plt.savefig(final_path)
plt.show()

ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr(warped, cls=True)
for line in result:
    for word_info in line:
        print(word_info[1][0])

# Define the output directory where the OCR result will be saved
output_text_dir = r"D:\Contour-Detection-OCR\ContourDetection\ocr_saved_text"

# Ensure the directory exists
os.makedirs(output_text_dir, exist_ok=True)

# Define the full path of the output text file
output_text_path = os.path.join(output_text_dir, f"{img_var}_ocr_result.txt")

# Open the file in write mode and properly format the output
with open(output_text_path, "w", encoding="utf-8") as f:
    for line in result:  # Iterate over detected lines
        if isinstance(line, list):  # Ensure it's a valid line
            words_in_line = [word_info[1][0] for word_info in line]  # Extract words
            line_text = " ".join(words_in_line)  # Join words to preserve spacing
            f.write(line_text.strip() + "\n")  # Write each line separately with a newline

print(f"OCR result saved to: {output_text_path}")


# result = ocr.ocr(warped_bw, cls=True)
# for line in result:
#     for word_info in line:
#         print(word_info[1][0])

#
# languages_list = ['en']
# reader = Reader(languages_list)
# results = reader.readtext(warped_bw)
# for r in results:
#     print(r[1])
# print('--------------------------------------------------------------')
# config_tesseract = "--tessdata-dir ./tessdata"
# result = pytesseract.image_to_string(erosion, lang='ron', config=config_tesseract)
# print(result)

# print('--------------------------------------------------------------')
# result2 = pytesseract.image_to_string(warped_bw, lang='ron', config=config_tesseract)
# print(result2)
#
# print('--------------------------------------------------------------')
# result3 = pytesseract.image_to_string(warped, lang='ron', config=config_tesseract)
# print(result3)



