import os
domain = 'economics'
target_model = 'gpt3'
model_evaluating = 'claude'
import pandas as pd
import torch
from typing import List

import tqdm
from idea_score_gpt import generate_openai
os.environ["OMP_NUM_THREADS"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder

# tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
# model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')

def text_to_list(text):
    # Split the text into individual lines
    topics = text.strip().split("\n")
    # Remove any empty strings resulting from splitting
    topics = [topic.strip() for topic in topics if topic.strip()]
    return topics

def cross_encoder(input_NLI):
    model = CrossEncoder("cross-encoder/nli-deberta-v3-base",  device="cpu")
    scores = model.predict(input_NLI)

    # Convert scores to labels
    label_mapping = ["contradiction", "entailment", "neutral"]
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    return labels

# def give_in_list(future_work):

#     future_work_list
#     return future_list

import re
matched_dic = {}
def infer_score_average(filename, pred: List[str], real: List[str]) -> float:
    """
    Computes the average entailment score between a predicate and a list of hypotheses.

    Args:
        hypotheses (List[str]): A list of hypothesis sentences.
        predicate (str): The predicate sentence.

    Returns:
        float: The average entailment score.
    """
    # Load the tokenizer and model
    response_list = []
    for real_item in real:
        prompt = f"""Your task is to examine whether a particular idea is incorporated within a set of ideas and to what degree. 
        This is a collection of ideas: {pred}
        This is a single idea: {real_item}
        Is the single idea contained within the collection of ideas?
        If yes, quantify its degree of presence or relevance of the single idea in the collection of ideas on a scale from 0 to 1.
        Output must me in the format example: (yes/no, score between 0 to 1)
        """
        response = generate_openai(prompt)
        response_list.append(response)

        # regex_pattern = r"([a-zA-Z]+)[^\d]+([\-]?\d*\.?\d+)"
        # matches = re.search(regex_pattern, response)
        # if matches:
        #     extracted_string = matches.group(1)
        #     extracted_float = float(matches.group(2))
        #     output = (extracted_string, extracted_float)
    matched_dic[filename] = {'predict_list':pred, 'real_list':real, 'response': response_list}

if domain == 'test':
    all_score = 0
    predicates = []
    hypothesiss = []
    path = 'data/Response2/' + model_name
    files = ['test']
    length = 0
    for file in files:
        if file == 'test':
            true_csv = pd.read_excel('data/Idea_'+file+'.xlsx')
        ids = true_csv['paper_id']
        future_work = true_csv['Future_work']
        result_dict = {ids[i]: future_work[i] for i in range(min(len(ids), len(future_work)))}
        text_paths  = os.listdir(os.path.join(path,file))
        for text_path in text_paths:
            if not text_path.endswith('.txt'):
                continue
            with open(os.path.join(os.path.join(path,file),text_path),'r') as f:
                print(text_path)
                text = f.read()
                real_future = text_to_list(text)
                if text_path.replace(".txt",'') in result_dict.keys():
                    predicate = result_dict[text_path.replace(".txt",'')]
                    score = infer_score_average(real_future,predicate)
                    all_score +=score
                    length+=1
                    
    print(all_score/length)

import json

if domain == 'economics':
    all_score = 0
    predicates = []
    hypothesiss = []
    path = 'data/Response2/' + target_model
    files = ['economics','eco_more']
    length = 0
    for file in files:
        if file == 'economics':
            true_csv = pd.read_excel('data/RealF/Idea_'+file+'.xlsx')
        else:
            true_csv = pd.read_excel('data/RealF/Idea_economicsMORE'+'.xlsx')
        ids = list(true_csv['paper_id'])
        future_work = true_csv['Response_Chat']
        
        #future_dict = {ids[i].split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}

        result_dict = {str(ids[i]).split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}
        text_paths  = os.listdir(os.path.join(path,file))
        for text_path in text_paths:
            if not text_path.endswith('.txt'):
                continue
            with open(os.path.join(os.path.join(path,file),text_path),'r') as f:
                print(text_path)
                text = f.read()
                hypothesis = text_to_list(text)
                if text_path.replace(".txt",'') in result_dict.keys():
                    predicate = result_dict[text_path.replace(".txt",'')]
                    predicate = text_to_list(predicate)
                    infer_score_average(text_path, hypothesis,predicate, model_evaluating)
                    # text_path.replace(".txt",'')
                    # all_score +=score
                    # length+=1
    dump_file_name = f'data/battle/{domain}'
    if not os.path.exists(dump_file_name):
        os.makedirs(dump_file_name)

    with open(f'{dump_file_name}/score_{target_model}vs{model_evaluating}.json', 'w') as f:
        json.dump(matched_dic,f)

if domain == 'computer':
    all_score = 0
    predicates = []
    hypothesiss = []
    path = 'data/Response2/' + model_name
    files = ['computer','comp_more']
    length = 0
    for file in files:
        if file == 'computer':
            true_csv = pd.read_excel('data/RealF/Idea_'+file+'.xlsx')
        else:
            true_csv = pd.read_excel('data/RealF/Idea_computerMORE'+'.xlsx')
        ids = list(true_csv['paper_id'])
        future_work = true_csv['Response_Chat']
        
        #future_dict = {ids[i].split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}

        result_dict = {str(ids[i]).split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}
        text_paths  = os.listdir(os.path.join(path,file))
        for text_path in text_paths:
            if not text_path.endswith('.txt'):
                continue
            with open(os.path.join(os.path.join(path,file),text_path),'r') as f:
                print(text_path)
                text = f.read()
                hypothesis = text_to_list(text)
                if text_path.replace(".txt",'') in result_dict.keys():
                    predicate = result_dict[text_path.replace(".txt",'')]
                    predicate = text_to_list(predicate)
                    infer_score_average(text_path, hypothesis,predicate)
                    # text_path.replace(".txt",'')
                    # all_score +=score
                    # length+=1


    dump_file_name = f'data/FRII_scores/{model_name}'
    if not os.path.exists(dump_file_name):
        os.makedirs(dump_file_name)

    with open(f'{dump_file_name}/score_{domain}.json', 'w') as f:
        json.dump(matched_dic,f)

if domain == 'medical':
    all_score = 0
    predicates = []
    hypothesiss = []
    path = 'data/Response2/' + model_name
    files = ['medical']
    length = 0
    for file in files:
        if file == 'medical':
            true_csv = pd.read_excel('data/RealF/Idea_'+file+'.xlsx')
        ids = list(true_csv['paper_id'])
        future_work = true_csv['Response_Chat']
        
        #future_dict = {ids[i].split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}
        result_dict = {str(ids[i]).split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}
        text_paths  = os.listdir(os.path.join(path,file))
        for text_path in text_paths:
            if not text_path.endswith('.txt'):
                continue
            with open(os.path.join(os.path.join(path,file),text_path),'r') as f:
                print(text_path)
                text = f.read()
                hypothesis = text_to_list(text)
                if text_path.replace(".txt",'') in result_dict.keys():
                    predicate = result_dict[text_path.replace(".txt",'')]
                    predicate = text_to_list(predicate)
                    infer_score_average(text_path, hypothesis,predicate)
                    # text_path.replace(".txt",'')
                    # all_score +=score
                    # length+=1


    dump_file_name = f'data/FRII_scores/{model_name}'
    if not os.path.exists(dump_file_name):
        os.makedirs(dump_file_name)

    with open(f'{dump_file_name}/score_{domain}.json', 'w') as f:
        json.dump(matched_dic,f)

if domain == 'physics':
    all_score = 0
    predicates = []
    hypothesiss = []
    path = 'data/Response2/' + model_name
    files = ['physics']
    length = 0
    for file in files:
        if file == 'physics':
            true_csv = pd.read_excel('data/RealF/Idea_'+file+'.xlsx')
        ids = list(true_csv['paper_id'])
        future_work = true_csv['Response_Chat']
        
        #future_dict = {ids[i].split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}
        result_dict = {str(ids[i]).split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}
        text_paths  = os.listdir(os.path.join(path,file))
        for text_path in text_paths:
            if not text_path.endswith('.txt'):
                continue
            with open(os.path.join(os.path.join(path,file),text_path),'r') as f:
                print(text_path)
                text = f.read()
                hypothesis = text_to_list(text)
                if text_path.replace(".txt",'') in result_dict.keys():
                    predicate = result_dict[text_path.replace(".txt",'')]
                    predicate = text_to_list(predicate)
                    infer_score_average(text_path, hypothesis,predicate)
                    # text_path.replace(".txt",'')
                    # all_score +=score
                    # length+=1


    dump_file_name = f'data/FRII_scores/{model_name}'
    if not os.path.exists(dump_file_name):
        os.makedirs(dump_file_name)

    with open(f'{dump_file_name}/score_{domain}.json', 'w') as f:
        json.dump(matched_dic,f)

import json
if domain == 'chemistry':
    all_score = 0
    predicates = []
    hypothesiss = []
    path = 'data/Response2/' + model_name
    files = ['chemistry', 'chem_more']
    length = 0
    for file in files:
        if file == 'chemistry':
            true_csv = pd.read_excel('data/RealF/Idea_'+file+'.xlsx')
        else:
            true_csv = pd.read_excel('data/RealF/Idea_chemistryMORE'+'.xlsx')
        ids = true_csv['paper_id']
        future_work = true_csv['Response_Chat']
        #future_dict = {ids[i].split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}

        result_dict = {ids[i].split('.')[0].replace('output_',''): future_work[i] for i in range(min(len(ids), len(future_work)))}

        text_paths  = os.listdir(os.path.join(path,file))
        for text_path in text_paths:
            if not text_path.endswith('.txt'):
                continue
            with open(os.path.join(os.path.join(path,file),text_path),'r') as f:
                print(text_path)
                text = f.read()
                hypothesis = text_to_list(text)
                if text_path.replace(".txt",'') in result_dict.keys():
                    predicate = result_dict[text_path.replace(".txt",'')]
                    predicate = text_to_list(predicate)
                    infer_score_average(text_path, hypothesis,predicate)
                    # text_path.replace(".txt",'')
                    # all_score +=score
                    # length+=1


    dump_file_name = f'data/FRII_scores/{model_name}'
    if not os.path.exists(dump_file_name):
        os.makedirs(dump_file_name)

    with open(f'{dump_file_name}/score_{domain}.json', 'w') as f:
        json.dump(matched_dic,f)



# if domain == 'computer':
#     all_score = 0
#     predicates = []
#     hypothesiss = []
#     path = 'data/Response2/' + model_name
#     files = ['comp_more','computer']
#     length = 0
#     for file in files:
#         if file == 'computer':
#             true_csv = pd.read_excel('data/Idea_'+file+'.xlsx')
#         else:
#             true_csv = pd.read_excel('data/Idea_computerMORE'+'.xlsx')
#         ids = true_csv['paper_id']
#         future_work = true_csv['Future_work']
#         result_dict = {str(ids[i]): future_work[i] for i in range(min(len(ids), len(future_work)))}

#         text_paths  = os.listdir(os.path.join(path,file))
#         for text_path in text_paths:
#             if not text_path.endswith('.txt'):
#                 continue
#             with open(os.path.join(os.path.join(path,file),text_path),'r') as f:
#                 print(text_path)
#                 text = f.read()
#                 hypothesis = text_to_list(text)
#                 if text_path.replace(".txt",'') in result_dict.keys():
#                     predicate = result_dict[text_path.replace(".txt",'')]
#                     score = infer_score_average(hypothesis,predicate)
#                     all_score +=score
#                     length+=1

#     print(all_score/length)


# if domain == 'economics':
#     all_score = 0
#     predicates = []
#     hypothesiss = []
#     path = 'data/Response/' + model_name
#     files = ['eco_more','economics']
#     length = 0
#     for file in files:
#         if file == 'economics':
#             true_csv = pd.read_excel('data/Idea_'+file+'.xlsx')
#         else:
#             true_csv = pd.read_excel('data/Idea_economicsMORE'+'.xlsx')
#         ids = true_csv['paper_id']
#         future_work = true_csv['Future_work']
#         result_dict = {str(ids[i]): future_work[i] for i in range(min(len(ids), len(future_work)))}

#         text_paths  = os.listdir(os.path.join(path,file))
#         for text_path in text_paths:
#             if not text_path.endswith('.txt'):
#                 continue
#             with open(os.path.join(os.path.join(path,file),text_path),'r') as f:
#                 print(text_path)
#                 text = f.read()
#                 hypothesis = text_to_list(text)
#                 if text_path.replace(".txt",'') in result_dict.keys():
#                     predicate = result_dict[text_path.replace(".txt",'')]
#                     score = infer_score_average(hypothesis,predicate)
#                     all_score +=score
#                     length+=1

#     print(all_score/length)