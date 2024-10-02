import re
import matplotlib.pyplot as plt

def parse_image_results(results_file):
    images = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    similarities = []
    overall_metrics = {}

    with open(results_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('Image:'):
            image_name = line.split(':')[1].strip()
            images.append(image_name)
        elif 'Accuracy:' in line and 'Overall' not in line:
            accuracy = float(line.split(':')[1].strip().replace('%', '')) / 100
            accuracies.append(accuracy)
        elif 'Precision:' in line and 'Overall' not in line:
            precision = float(line.split(':')[1].strip().replace('%', '')) / 100
            precisions.append(precision)
        elif 'Recall:' in line and 'Overall' not in line:
            recall = float(line.split(':')[1].strip().replace('%', '')) / 100
            recalls.append(recall)
        elif 'F1-Score:' in line and 'Overall' not in line:
            f1_score = float(line.split(':')[1].strip().replace('%', '')) / 100
            f1_scores.append(f1_score)
        elif 'Similarity:' in line:
            similarity = float(line.split(':')[1].strip().replace('%', '')) / 100
            similarities.append(similarity)
        elif 'Overall Accuracy:' in line:
            overall_metrics['Accuracy'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Overall Precision:' in line:
            overall_metrics['Precision'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Overall Recall:' in line:
            overall_metrics['Recall'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Overall F1-Score:' in line:
            overall_metrics['F1-Score'] = float(line.split(':')[1].strip().replace('%', ''))

    results = list(zip(images, accuracies, precisions, recalls, f1_scores, similarities))
    sorted_results = sorted(results, key=lambda x: int(re.search(r'img(\d+)', x[0]).group(1)))

    images, accuracies, precisions, recalls, f1_scores, similarities = zip(*sorted_results)

    return images, accuracies, precisions, recalls, f1_scores, similarities, overall_metrics

def plot_image_results(ocr_tools_results, metric_name, metric_index):
    fig, ax = plt.subplots(figsize=(10, 6))

    for tool_name, results in ocr_tools_results.items():
        images, accuracies, precisions, recalls, f1_scores, similarities, _ = results
        metrics = [accuracies, precisions, recalls, f1_scores, similarities][metric_index]

        if len(images) != len(metrics):
            print(f"Warning: Mismatch in lengths for {tool_name}. Truncating to the shortest list.")
            min_length = min(len(images), len(metrics))
            images = images[:min_length]
            metrics = metrics[:min_length]

        x_values = list(range(1, len(images) + 1))

        ax.plot(x_values, metrics, label=tool_name, marker='o')

    ax.set_xticks(range(1, 16))
    ax.set_xticklabels(range(1, 16))

    ax.set_title(f'{metric_name} Comparison Across Images')
    ax.set_xlabel('Image Number')
    ax.set_ylabel(f'{metric_name}')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_overall_results(ocr_tools_results):
    overall_data = {'Tool': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

    for tool_name, results in ocr_tools_results.items():
        _, _, _, _, _, _, overall_metrics = results
        overall_data['Tool'].append(tool_name)
        overall_data['Accuracy'].append(overall_metrics['Accuracy'])
        overall_data['Precision'].append(overall_metrics['Precision'])
        overall_data['Recall'].append(overall_metrics['Recall'])
        overall_data['F1-Score'].append(overall_metrics['F1-Score'])

    fig, ax = plt.subplots(figsize=(10, 6))

    tool_names = overall_data['Tool']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bar_width = 0.2  # Width of the bars

    for i, metric in enumerate(metrics):
        bar_positions = [x + i * bar_width for x in range(len(tool_names))]
        bars = ax.bar(bar_positions, overall_data[metric], width=bar_width, label=metric)

        for bar, tool_name in zip(bars, tool_names):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{tool_name}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks([x + bar_width * 1.5 for x in range(len(tool_names))])
    ax.set_xticklabels(tool_names)

    ax.set_title('Overall Metrics Comparison')
    ax.set_xlabel('OCR Tool')
    ax.set_ylabel('Percentage')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

ocr_tools = {
    'EasyOCR': 'easyocr_results.txt',
    'Keras OCR': 'keras_ocr_results.txt',
    'Tesseract': 'tesseract_results.txt'
}

ocr_tools_results = {}
for tool_name, results_file in ocr_tools.items():
    images, accuracies, precisions, recalls, f1_scores, similarities, overall_metrics = parse_image_results(results_file)
    ocr_tools_results[tool_name] = (images, accuracies, precisions, recalls, f1_scores, similarities, overall_metrics)

plot_image_results(ocr_tools_results, 'Accuracy', 0)
plot_image_results(ocr_tools_results, 'Precision', 1)
plot_image_results(ocr_tools_results, 'Recall', 2)
plot_image_results(ocr_tools_results, 'F1-Score', 3)
plot_image_results(ocr_tools_results, 'Similarity', 4)

plot_overall_results(ocr_tools_results)
