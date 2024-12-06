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
from code.llama import generate_llama
#from gemini import inference as gemini_inference  #keep changing
#from claude import inference as claude_inference
#from galactica import inference as galactica_inference

#from gpt import generate_openai

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

def generate_gpt(paper_text):
     prompt = f"""Imagine you are a research scientist. After reading the following paper, brainstorm to generate potential future research ideas.:  
        ``` {paper_text}```
         Potential future research ideas from the paper in bullet points are: """
     response = generate_openai(prompt)
     return response 

#Change this for each models
def generate_gemini(paper_text):
    prompt = f"""Imagine you are a research scientist. After reading the following paper, brainstorm to generate potential future research ideas.:  
        ``` {paper_text}```
        Potential future research ideas from the paper in bullet points are: """

    response = gemini_inference(prompt)
    return response

def generate_claude2(paper_text):
    prompt = f"""{HUMAN_PROMPT} Imagine you are a research scientist. After reading the following paper, brainstorm to generate all potential key future research ideas. :  
        {paper_text}
        Potential future research ideas from the paper in bullet points are: {AI_PROMPT}"""
    response = claude_inference(prompt)
    return response

def generate_galactica(paper_text):
    prompt = f""" Imagine you are a research scientist, read the following paper and generate all potential future research ideas after brainstorming: 
        {paper_text[0:1000]+paper_text[-1000:]}
        5 possible future research ideas from the paper are: <work>:"""
    response = galactica_inference(prompt)
    return response

def choose_model(text):
    if model_name == 'gemini':
        return generate_gemini(text)
    elif model_name == 'claude':
        return generate_claude2(text)
    elif model_name == 'gpt4':
        return generate_gpt(text)
    elif model_name=='llama':
        return generate_llama_70b(text)

for filename in tqdm(os.listdir(input_folder)):
    if not filename.endswith('.txt'):
        continue
    dump_filename = os.path.join(dump_folder, filename)

    if os.path.exists(dump_filename):
            continue
    
    reponse = ''
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