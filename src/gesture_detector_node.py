#!/usr/bin/env python3

import os
import requests
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline

print("HII")
print(os.getcwd())

image_url = "https://llava-vl.github.io/static/images/view.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

model_id = "llava-hf/llava-1.5-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})


max_new_tokens = 200
prompt = "USER: <image>\nWhat are the things I should be cautious about when I visit this place?\nASSISTANT:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

print(outputs[0]["generated_text"])