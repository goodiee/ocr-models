# Project: Comparison of OCR Tools with Preprocessing Techniques

Overview
This project focuses on comparing three OCR tools—EasyOCR, TesseractOCR, and Keras-OCR—with potential improvements through preprocessing techniques applied using OpenCV. The goal is to assess the performance of these OCR models and explore how preprocessing can enhance accuracy, especially for a diverse vision dataset of images.

Since the images in the dataset vary significantly in characteristics, we categorized them into subsets. Applying a single preprocessing technique to all images would not yield optimal results, so a manual labeling tool was created to assign appropriate labels to each image, allowing targeted preprocessing. The labeled images are stored in image_labels.json, which streamlines the process by associating each image with its corresponding preprocessing method.
