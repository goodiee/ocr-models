import os
import cv2
import json
import matplotlib.pyplot as plt

image_folder = 'vision_datasets'
output_directory = 'image_labels'
json_file_path = os.path.join(output_directory, 'image_labels.json')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

image_files = os.listdir(image_folder)
image_labels = {}


def show_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()


for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    show_image(image_path)

    print("Select one or more labels for the image (separate numbers with a space):")
    print("1: Blur-High-Contrast")
    print("2: Handwriting")
    print("3: White Background")
    print("4: Same Back Stylewriting")
    print("5: White on Black")

    labels_input = input("Enter label numbers separated by space: ")

    label_numbers = labels_input.split()

    labels = []
    for number in label_numbers:
        if number == '1':
            labels.append('Blur-High-Contrast')
        elif number == '2':
            labels.append('Handwriting')
        elif number == '3':
            labels.append('White Background')
        elif number == '4':
            labels.append('Same Back Stylewriting')
        elif number == '5':
            labels.append('White on Black')

    image_labels[image_file] = labels

with open(json_file_path, 'w') as json_file:
    json.dump(image_labels, json_file, indent=4)

print(f"Labels saved in file {json_file_path}")
