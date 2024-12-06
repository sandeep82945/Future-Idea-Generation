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
import openai
from tqdm import tqdm
#from code.llama import generate_llama


with open('/mnt/d/Future-Idea-Generation/config2.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

paper_dump_path = config["paper_dump_path"]
input_folder = config["input_folder"]
model_name = config["model_name"]
domain = input_folder.split('/')[-1]

dump_folder = os.path.join(paper_dump_path,model_name, domain)
if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)


def generate_llama_70b(paper_text):
     prompt = f"""Imagine you are a research scientist. After reading the following paper, brainstorm to generate potential future research ideas.:  
        ``` {paper_text}```
         Potential future research ideas from the paper in bullet points are: """
     response = generate_llama(prompt)
     return response 


def generate_llama(prompt):
    model_id = "meta-llama/Meta-Llama-3.1-70B"
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return (outputs[0]["generated_text"][-1])


def choose_model(text):
    return generate_llama_70b(text)

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

