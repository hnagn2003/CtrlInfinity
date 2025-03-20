import cv2
import json
import os
from PIL import Image
save_folder = "data/infinity_toy_data/condition_canny"
image_folder = "data/infinity_toy_data/images"
os.makedirs(save_folder, exist_ok=True)
new_data_to_dump = []
for file in os.listdir(image_folder):
    image_name = file.split('/')[-1]
    image_path = os.path.join(image_folder, file)
    image = cv2.imread(image_path)
    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    Image.fromarray(image).save(os.path.join(save_folder, image_name))
