import os
import cv2
import json
import numpy as np

image_folder = 'vision_datasets'
processed_folder = 'processed_images'
json_file_path = os.path.join('image_labels', 'image_labels.json')

if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)


def preprocess_image_same_back_stylewriting(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    return enhanced_image


def preprocess_handwritten_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    blurred = cv2.medianBlur(image, 5)

    # Convert image to LAB for contrast adjustment
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)

    # Merge channels back
    enhanced_lab = cv2.merge((enhanced_l, a, b))

    # Convert back to BGR format
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image


def preprocess_white_on_black_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, 0)

    # Enhance contrast
    contrast_img = cv2.convertScaleAbs(img, alpha=1.0, beta=0)

    # Sharpen the image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp_img = cv2.filter2D(contrast_img, -1, kernel)

    # Invert colors to highlight white text on a black background
    inverted_img = cv2.bitwise_not(sharp_img)

    return inverted_img


def preprocess_blur_high_contrast_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Adjust contrast and brightness
    alpha = 0.7
    beta = -90

    # Apply brightness and contrast adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Apply a filter to improve sharpness
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(adjusted_image, -1, kernel)

    return sharpened


def preprocess_white_background_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply simple thresholding (binarization)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh


with open(json_file_path, 'r') as json_file:
    image_labels = json.load(json_file)

# Apply preprocessing based on the label
for image_file, labels in image_labels.items():
    image_path = os.path.join(image_folder, image_file)

    if 'Same Back Stylewriting' in labels:
        processed_image = preprocess_image_same_back_stylewriting(image_path)
        output_label = 'Same Back Stylewriting'
    elif 'Handwriting' in labels:
        processed_image = preprocess_handwritten_image(image_path)
        output_label = 'Handwriting'
    elif 'White on Black' in labels:
        processed_image = preprocess_white_on_black_image(image_path)
        output_label = 'White on Black'
    elif 'Blur-High-Contrast' in labels:
        processed_image = preprocess_blur_high_contrast_image(image_path)
        output_label = 'Blur-High-Contrast'
    elif 'White Background' in labels:
        processed_image = preprocess_white_background_image(image_path)
        output_label = 'White Background'
    else:
        print(f"Skipping {image_file}, no relevant label found.")
        continue

    if processed_image is not None:
        output_path = os.path.join(processed_folder, image_file)
        cv2.imwrite(output_path, processed_image)
        print(f"Processed and saved {output_label} labeled image: {output_path}")
    else:
        print(f"Processing failed for {image_file} with label {output_label}")
