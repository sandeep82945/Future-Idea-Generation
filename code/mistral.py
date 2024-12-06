import transformers
import torch

import os
import pandas as pd
import json
import nltk
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
nltk.download('punkt')
import yaml
tokenizer = PunktSentenceTokenizer()
from test_preprocessing import preprocess
from anthropic_bedrock import AnthropicBedrock, HUMAN_PROMPT, AI_PROMPT
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer



with open('/mnt/d/Future-Idea-Generation/config2.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

paper_dump_path = config["paper_dump_path"]
input_folder = config["input_folder"]
model_name = config["model_name"]
domain = input_folder.split('/')[-1]

dump_folder = os.path.join(paper_dump_path,model_name, domain)
if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)

def generate_mixtral(paper_text):
     prompt = f"""Imagine you are a research scientist. After reading the following paper, brainstorm to generate potential future research ideas.:  
        ``` {paper_text}```
         Potential future research ideas from the paper in bullet points are: """
     response = generate_mixtral_8(prompt)
     return response 

def generate_mixtral_8(prompt):
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id)

    text = prompt
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=20)
    return (tokenizer.decode(outputs[0], skip_special_tokens=True))


def choose_model(text):
    return generate_mixtral(text)

for filename in tqdm(os.listdir(input_folder)):
    if not filename.endswith('.txt'):
        continue
    dump_filename = os.path.join(dump_folder, filename)

    if os.path.exists(dump_filename):
            continue
    
    response = ''
    filepath = os.path.join(input_folder,filename)
    with open(filepath, 'r') as f:
         text = f.read()
         text = text.replace('\n', ' ')
         response = preprocess(text)
         response = choose_model(response)
         if response is None:
             continue

    with open(dump_filename, 'w') as f:
        f.write(response)