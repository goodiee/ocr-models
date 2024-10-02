import os
import easyocr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from difflib import SequenceMatcher
import re

preprocessed_images_folder = 'processed_images/'
ground_truth_file = 'ground_truth'
results_file = 'results/easyocr_results.txt'

reader = easyocr.Reader(['en'], gpu=False)


def load_ground_truth(ground_truth_file):
    ground_truth = {}
    current_image = None
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                current_image = line[2:].strip()
                ground_truth[current_image] = ""
            elif current_image:
                ground_truth[current_image] += line + "\n"

    for image in ground_truth:
        ground_truth[image] = ground_truth[image].strip()

    return ground_truth


def extract_chars(text):
    return re.sub(r'\s+', '', text)


def calculate_similarity(ocr_text, ground_truth_text):
    ocr_text = extract_chars(ocr_text.lower())
    ground_truth_text = extract_chars(ground_truth_text.lower())

    return SequenceMatcher(None, ocr_text, ground_truth_text).ratio()


def calculate_metrics(ocr_text, ground_truth_text):
    ocr_chars = list(extract_chars(ocr_text.lower()))
    ground_truth_chars = list(extract_chars(ground_truth_text.lower()))

    min_len = min(len(ocr_chars), len(ground_truth_chars))
    ocr_chars = ocr_chars[:min_len]
    ground_truth_chars = ground_truth_chars[:min_len]
    max_len = max(len(ocr_chars), len(ground_truth_chars))

    if max_len == 0:
        return 0, 0, 0, 0

    accuracy = accuracy_score(ground_truth_chars, ocr_chars)
    precision = precision_score(ground_truth_chars, ocr_chars, average='macro', zero_division=1)
    recall = recall_score(ground_truth_chars, ocr_chars, average='macro', zero_division=1)
    f1 = f1_score(ground_truth_chars, ocr_chars, average='macro', zero_division=1)

    return accuracy, precision, recall, f1


def perform_ocr_evaluation(images_folder, ground_truth_data, results_file):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_images = 0

    with open(results_file, 'w', encoding='utf-8') as f:
        for image_file in os.listdir(images_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):

                image_path = os.path.join(images_folder, image_file)

                if not os.path.exists(image_path):
                    output = f"Image not found: {image_path}. Skipping...\n"
                    print(output)
                    f.write(output)
                    continue

                if image_file in ground_truth_data:
                    ground_truth_text = ground_truth_data[image_file]

                    try:
                        result = reader.readtext(image_path, detail=0)
                        ocr_text = " ".join(result).replace('\n', ' ')
                    except Exception as e:
                        output = f"Error processing {image_file}: {e}\n"
                        print(output)
                        f.write(output)
                        continue

                    similarity = calculate_similarity(ocr_text, ground_truth_text)
                    accuracy, precision, recall, f1 = calculate_metrics(ocr_text, ground_truth_text)

                    output = (
                        f"Image: {image_file}\n"
                        f"OCR Result: {ocr_text}\n"
                        f"Ground Truth: {ground_truth_text}\n"
                        f"Similarity: {similarity * 100:.2f}%\n"
                        f"Accuracy: {accuracy * 100:.2f}%\n"
                        f"Precision: {precision * 100:.2f}%\n"
                        f"Recall: {recall * 100:.2f}%\n"
                        f"F1-Score: {f1 * 100:.2f}%\n"
                        + "-" * 40 + "\n"
                    )

                    print(output)
                    f.write(output)

                    total_accuracy += accuracy
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1
                    total_images += 1
                else:
                    output = f"No ground truth found for {image_file}. Skipping...\n"
                    print(output)
                    f.write(output)

        if total_images > 0:
            overall_accuracy = total_accuracy / total_images
            overall_precision = total_precision / total_images
            overall_recall = total_recall / total_images
            overall_f1 = total_f1 / total_images

            overall_output = (
                f"Overall Accuracy: {overall_accuracy * 100:.2f}%\n"
                f"Overall Precision: {overall_precision * 100:.2f}%\n"
                f"Overall Recall: {overall_recall * 100:.2f}%\n"
                f"Overall F1-Score: {overall_f1 * 100:.2f}%\n"
            )
            print(overall_output)
            f.write(overall_output)
        else:
            print("No images processed.")
            f.write("No images processed.\n")


ground_truth_data = load_ground_truth(ground_truth_file)
perform_ocr_evaluation(preprocessed_images_folder, ground_truth_data, results_file)
