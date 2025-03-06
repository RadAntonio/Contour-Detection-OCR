import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re
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
    bw_image = cv2.adaptiveThreshold(blurred, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 10)
    return bw_image


def draw_text_boxes(image, ocr_results):
    """
    Draw bounding boxes and recognized text on the image.
    """
    annotated_image = image.copy()
    for line in ocr_results:
        for word_info in line:
            box, (text, confidence) = word_info
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # Draw recognized text above the top-left point of the box
            cv2.putText(annotated_image, text, (int(box[0][0]), int(box[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return annotated_image


# Regex to detect price tokens (e.g., 12.40, optionally with a trailing letter)
price_pattern = re.compile(r"\d+\.\d{2}[A-Z]?", re.IGNORECASE)


def associate_products_with_prices(line_text):
    """
    Splits a line into tokens, then uses regex to identify price tokens.
    Returns a tuple: (product description, price)
    """
    tokens = line_text.split()
    product_tokens = []
    price_tokens = []

    for token in tokens:
        if price_pattern.fullmatch(token):
            price_tokens.append(token)
        else:
            product_tokens.append(token)

    product = " ".join(product_tokens)
    price = " ".join(price_tokens)
    return product, price


# ------------------ Main Processing ------------------

output_dir = './saved_images'
img_var = 'test32'
image_path = rf'./images/{img_var}.jpg'

# Read and preprocess image
image1 = cv2.imread(image_path)
img_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
otsu_threshold, binary_image = cv2.threshold(blurred, 0, 255,
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Optional: Display intermediate images
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

# Find contours and draw them
contours1, hierarchy1 = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
image_contours1 = cv2.drawContours(image1.copy(), contours1, -1, (255, 255, 0), 2, cv2.LINE_AA)
plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(image_contours1, cv2.COLOR_BGR2RGB))
plt.title('Contour Image')

# Find the largest contour (assumed to be the receipt)
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

    erosion = cv2.erode(warped_bw, np.ones((2, 2), np.uint8))
    plt.subplot(2, 4, 8)
    plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
    plt.title('Erosion Image')
else:
    print("Could not find a quadrilateral contour for the receipt.")

plt.tight_layout()
final_path = os.path.join(output_dir, f'{img_var}_final_plot.png')
plt.savefig(final_path)
plt.show()

# ------------------ OCR and Text Extraction ------------------

ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr(warped, cls=True)

# Print OCR text to console and save grouped text lines
ocr_lines = []
for line in result:
    if isinstance(line, list):
        words_in_line = [word_info[1][0] for word_info in line]
        line_text = " ".join(words_in_line)
        ocr_lines.append(line_text)
        print(line_text)

output_text_dir = r"D:\Contour-Detection-OCR\ContourDetection\ocr_saved_text"
os.makedirs(output_text_dir, exist_ok=True)
output_text_path = os.path.join(output_text_dir, f"{img_var}_ocr_result.txt")
with open(output_text_path, "w", encoding="utf-8") as f:
    for line_text in ocr_lines:
        f.write(line_text.strip() + "\n")
print(f"OCR result saved to: {output_text_path}")

# ------------------ Product-Price Association ------------------

# Process each OCR line to associate product and price
associated_items = []
for line_text in ocr_lines:
    product, price = associate_products_with_prices(line_text)
    associated_items.append({"product": product, "price": price})

# Print the associations
print("\nAssociated Product-Price pairs:")
for item in associated_items:
    print(f"Product: {item['product']} | Price: {item['price']}")

# ------------------ Draw Text Boxes on Warped Image ------------------

annotated_image = draw_text_boxes(warped, result)
annotated_output_path = os.path.join(output_dir, f"{img_var}_annotated.png")
cv2.imwrite(annotated_output_path, annotated_image)
print(f"Annotated image saved to: {annotated_output_path}")
