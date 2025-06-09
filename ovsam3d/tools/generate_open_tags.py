import argparse
import os

import numpy as np
import random
import openai
import torch
from tqdm import tqdm
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from ovsam3d.data.load import Images, get_number_of_images
from ram.utils import build_openset_llm_label_embedding
import json
from torch import nn

parser = argparse.ArgumentParser(
    description='Generate open tags for any scene from RAM')
parser.add_argument('--dataset_path',
                    metavar='DIR',
                    help='path to dataset',
                    default='/data/ScanNet/scans')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='ram_checkpoints/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')
parser.add_argument('--output_path',
                    metavar='DIR',
                    help='path to LLM tag descriptions',
                    default='/output/scannet200_text')


def filter_tags(tags):
    # Generate LLM tag descriptions
    # FIXME: You can choose more precise prompts to better filter out instance tags.
    llm_prompt = f"I want you to act as an English expert in words understanding.\
    Please understand every word in the word lists:\n{tags}\n Select appropriate words to form a new word list as following rules:\n \
    -the word should refer to a specific object such as (chair, table, television, etc.), delete ambiguous things such as(appliance, furniture, etc.);\n \
    -the word should not represent colors, like (brown, white, etc.);\n \
    -the word should not refer to room types or some scene, like (bathroom, living room, street, cinema, etc.); \n \
    -if there are some words with similar meanings, please keep the most general term and delete others, for example, retaining 'table' and \n \
    delete 'kitchen table', 'glass table' in (table, kitchen table, glass table); \n \
    Please recognize the words from the provided word lists according to the above rules, then output the final selected word list without dialog."

    filter_tag = []
    # send message
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "assistant", "content": llm_prompt}],
            max_tokens=2000,
            temperature=0.99,
            stop=None
        )
    except Exception as e:
        print(e)
        return []

    # parse the response
    for item in response.choices:
        filter_tag = item.message['content'].strip()
    return filter_tag


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    # Load model
    model = ram_plus(pretrained=args.pretrained,
                     image_size=args.image_size,
                     vit='swin_l')
    model.eval()
    model = model.to(device)

    # Adding closed-set categories to the open category library could yield better results, 
    # but this contradicts the open-world setting. 
    # Therefore, we choose to extract tags directly from the RAM category library.
    dataset_path = args.dataset_path
    scans = os.listdir(dataset_path)
    # Load all images in a scene based on the frequency
    for scan in scans:
        if os.path.exists(os.path.join(args.output_path, scan+'_text.txt')):
            continue
        indices = np.arange(0, get_number_of_images(os.path.join(dataset_path, scan, 'data/pose')), step=10)
        images = Images(images_path=os.path.join(dataset_path, scan, 'data_compressed/color'),
                        extension=".jpg",
                        indices=indices)
        print(f"[INFO] Images loaded. {len(images.images)} images found.")
        # Run RAM
        image_tags = []
        for image in images.images:
            image = transform(image).unsqueeze(0).to(device)
            res = inference(image, model)
            tags = res[0].split(" | ")
            image_tags = image_tags + tags
        image_tags = np.array(image_tags)

        # Filter unique assembled tags through ChatGPT
        image_tag, tags_num = np.unique(image_tags, return_counts=True)

        # Set OpenAI API key
        openai.api_key = ''
        openai.api_base = ''
        select_tags = filter_tags('\n'.join(image_tag))
        if len(select_tags) == 0:
            continue
        # Save open tags as .txt file
        with open(os.path.join(args.output_path, scan+'_text.txt'), 'w') as file:
            file.write(select_tags)
