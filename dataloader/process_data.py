image_folder = '../RepControlNet/data/canny_laion/laion10k/train'
json_path = '../RepControlNet/data/canny_laion/laion10k/text_embedding_10k.json'
longcap_folder = '../RepControlNet/data/canny_laion/laion10k/long_caption'
save_path = '../RepControlNet/data/canny_laion/infinity_10k/splits/1.000_000010000.jsonl'
# read all data from json in longcap_folder
import json
import os
from tqdm import tqdm
long_prompts = {}
for file in os.listdir(longcap_folder):
    with open(os.path.join(longcap_folder, file), 'r') as f:
        data = json.load(f)
        long_prompts.update(data)
data_to_save = []
# read all data from json_path
file = open(save_path, 'a')
with open(json_path, 'r') as f:
    data = json.load(f)
    for sample in tqdm(data):
        image_name = sample['image_path']
        image_path = os.path.join(image_folder, image_name)
        short_prompt=sample['text']
        long_prompt = long_prompts[image_name]
        data_to_save = {'image_name': image_name, 'text': short_prompt, 'long_caption': long_prompt, 'long_caption_type': 'caption-InternVL2.0', 'short_caption_type': 'laion', 'h_div_w': 1.0}
        file.write(json.dumps(data_to_save) + '\n')
file.close() 