import os
import random
import shutil
from dicts import cat_dicts
import re
from classification_states import a

# Define the path to your folder of folders of images
base_folder = "object-states/ChangeIT-Subset-Images/"

# Define the percentage of images you want to select
percentage = 100

categories =['apple', 'avocado', 'bacon', 'ball', 'beer', 'boil_milk', 'box', 'butter', 'cake', 'candle', 'champagne', 'cherries', 'chocolate', 'coffee', 'corn', 'cream', 'cube', 'dough', 'dragon_fruit', 'drill', 'eggs', 'fish', 'garlic', 'grill', 'juice', 'milk', 'onion', 'outlet', 'pan', 'pancake', 'pineapple', 'plane', 'potatoes', 'ribbon', 'rope', 'shoes', 't-shirt', 'tea', 'tie', 'tile', 'tortilla', 'tree', 'weed', 'whisk_eggs']

def get_folder_names(directory):
    """Get all folder names within a directory."""
    folder_names = [folder_name for folder_name in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder_name))]
    return folder_names
def extract_frame_number(file_name):
    """Extracts the frame number from a file name."""
    k = int(file_name.split("_")[1].split(".")[0])
    print(file_name)
    print(k)
    return (k-1)

def filter_labels(labels, selected_images_folder):
    """Filters a list of labels based on the selected images in the given folder."""
    selected_frames = []
    for file_name in os.listdir(selected_images_folder):
        if file_name.endswith(".txt"):
            continue
        selected_frames.append(extract_frame_number(file_name))
    print(selected_frames)
    filtered_labels = [labels[frame] for frame in selected_frames]
    return filtered_labels, selected_frames

def get_image_files(folder_path):
    """Get a list of image file names in the given folder."""
    image_files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            if file_name.lower().endswith('.jpg'):
                image_files.append(file_name)
            elif file_name.lower().endswith('.txt'):
                os.remove(file_path)  # Remove the text file
    return image_files

def filename_to_index(filename):
    # Extract the numeric part from the filename
    number_str = re.search(r'\d+', filename).group()
    # Convert to integer and subtract 1
    return int(number_str) - 1

folder_names = get_folder_names(base_folder)

annotations = {}
for category in categories:
        # print(category)
        image_counter = 0
        for videos in cat_dicts[category]:
            video_name = videos[0]
            video_annotation = videos[1]
            # print(video_name)
            if video_name in folder_names:
                directory_path = base_folder + video_name
                frame_array = get_image_files(directory_path)
                gt = list(video_annotation.values())
                for frame in frame_array:
                    fl_name = video_name + "_" + frame
                    annotations[fl_name] = (gt[filename_to_index(frame)], category)

