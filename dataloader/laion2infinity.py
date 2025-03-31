json_path = '../RepControlNet/data/canny_laion/text_embedding.json'
save_path = '../RepControlNet/data/Infinity10k/splits/1.000_000020000.jsonl'
# read all data from json in longcap_folder
import json
import os
from tqdm import tqdm
data_to_save = []
# read all data from json_path
file = open(save_path, 'w')
with open(json_path, 'r') as f:
    data = json.load(f)[:20000]
    for sample in tqdm(data):
        image_name = sample['image_path']
        short_prompt=sample['text']
        long_prompt = sample['text']
        data_to_save = {'image_name': image_name, 'text': short_prompt, 'long_caption': long_prompt, 'long_caption_type': 'caption-InternVL2.0', 'short_caption_type': 'laion', 'h_div_w': 1.0}
        file.write(json.dumps(data_to_save) + '\n')
file.close() 