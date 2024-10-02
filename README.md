# Project: Comparison of OCR Tools with Preprocessing Techniques

## Overview
This project focuses on comparing three OCR tools—EasyOCR, TesseractOCR, and Keras-OCR—with potential improvements through preprocessing techniques applied using OpenCV. The goal is to assess the performance of these OCR models and explore how preprocessing can enhance accuracy, especially for a diverse vision dataset of images.

Since the images in the dataset are quite different, it was decided to categorize them into subsets based on common features. Because applying a single preprocessing technique to all images would not be optimal results, so a manual labeling tool was created to assign appropriate labels to each image, allowing targeted preprocessing. The labeled images are stored in `image_labels.json`, which streamlines the process by associating each image with its corresponding preprocessing method. It is manual tool since there is a not large dataset we have. However, it is still good tool to speed up the process. 

## Project Structure
### Preprocessing Based on Image Labels

The following preprocessing techniques are applied based on the assigned labels in the dataset:

- Same Back Stylewriting:

`preprocess_image_same_back_stylewriting`
Description: Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

- Handwriting:

`preprocess_handwritten_image()`
Description: Reduces noise using median blurring, converts to LAB color space for contrast adjustment (CLAHE on the L channel), and converts back to BGR format.

- White on Black:

`preprocess_white_on_black_image()`
Description: Enhances contrast, sharpens the image, and inverts colors to make white text on a black background more visible.

- Blur-High-Contrast:

`preprocess_blur_high_contrast_image()`
Description: Adjusts brightness and contrast and applies sharpening filters to improve clarity.

- White Background:

`preprocess_white_background_image()`
Description: Converts the image to grayscale and applies simple thresholding (binarization) to make text on white backgrounds clearer.

Processed images are saved in the *processed_images/* folder, and the appropriate preprocessing technique is applied based on the image's label.

! The list of labels can be expaneded if needed depending on the dataset. Also it is possible to apply some tecniques for one image.

### OCR Tools and Evaluation

After preprocessing, the OCR tools are applied:

- EasyOCR `easy-ocr.py`: Requires additional files (already included in the project) for setup. GPU is recommended for faster processing.

- Keras-OCR `keras-ocr.py`: Also works better with GPU acceleration.

- Tesseract-OCR `tesseract-ocr.py`: No extra installation is required beyond the basic setup.

The results from each OCR tool are stored in their respective text files within the *results/* folder. These text files contain similarity metrics such as accuracy, precision, recall, and F1-score for each image, along with the overall results at the end.

### Ground Truth and Evaluation Script
A ground truth file `ground-truth.txt` is provided for evaluating OCR output accuracy by comparing the original text with the results produced by each OCR tool.

For detailed analysis, the `results.py` script can be run to generate plots showing performance dynamics for each image. This script helps visualize how different preprocessing techniques and OCR tools compare in accuracy across the dataset.

## Setup Instructions

1. Install the required dependencies using:

`pip install -r requirements.txt`

2. Run the labeling tool to manually label the images:

`python labeling-tool.py`

3. Run the preprocessing script to apply preprocessing techniques based on image labels:

`python preprocessing.py`

4. Once the images are preprocessed, run the OCR tools:

`python easy-ocr.py
python keras-ocr.py
python tesseract-ocr.py
`

5. Check the results in the *results/* folder. Each OCR tool will output results, including metrics like similarity, accuracy, precision, recall, and F1-score.

6. For further analysis, run the following command to generate performance plots:

`python results.py`

## Results Summary

EasyOCR showed the highest accuracy among the three OCR tools accrding to plots below. The x axis - is number of image, y - metric measure. 
Preprocessing improved accuracy by around 10-12%, though primarily for EasyOCR. It is because of complexities of dataset and needs to improve preprocessing. 

![1](https://github.com/user-attachments/assets/3a8a3857-bb17-422f-bc34-92c65e90d64a)

![2](https://github.com/user-attachments/assets/3e08e66b-c3f4-4af5-908d-1ad652aa6345)

![3](https://github.com/user-attachments/assets/ebaf3353-3ec4-401b-aca4-17f666481d6d)

![4](https://github.com/user-attachments/assets/b4b00cf7-2fdf-457f-ba88-9a41a733fdb3)

![image](https://github.com/user-attachments/assets/772b16d1-43c8-4091-bc6b-d7733c973c0a)


## Cited Resources

Tesseract OCR Documentation: https://github.com/UB-Mannheim/tesseract/wiki

EasyOCR: A Comprehensive Guide: https://medium.com/@adityamahajan.work/easyocr-a-comprehensive-guide-5ff1cb850168

KesarOCR documentation: https://keras-ocr.readthedocs.io/en/latest/
