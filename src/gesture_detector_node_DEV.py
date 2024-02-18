#!/usr/bin/env python3

import os
import requests
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline

print(os.getcwd())

# image_url = "https://llava-vl.github.io/static/images/view.jpg"
# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

model_id = "llava-hf/llava-1.5-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})


max_new_tokens = 200

description = "lady in scarf"

# prompt = f"USER: <image>\n You are a system that decides which person to follow. \
#     The user has inputted the following description: {description}. \
#     If this description matches the description of a person within the image please output the bounding box. \
#     If no you don't find something in the image that closely resembles this description please output None. \
#     If the description says something that instructs you to stop following people please output None. \
#     \nASSISTANT:"

prompt = f"USER: <image>\n Does the following description exist in the image? Description: {description}. If so please output the bounding box the person that best matches the description. Otherwise, if not please explain why you don't see a person as described by the user input. \
    \nASSISTANT:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

print(outputs[0]["generated_text"])