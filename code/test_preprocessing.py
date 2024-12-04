import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import string
import os
import re

nltk.download('punkt')

import re

def remove_websites(text):
    """
    Removes website URLs, including those without standard 'http://' or 'https://' prefixes,
    from the provided text.
    
    Args:
    text (str): The text from which to remove website references.
    
    Returns:
    str: The text with website references removed.
    """
    # Remove standard URLs
    no_urls = re.sub(r'https?:\/\/\S+', '', text)
    
    # Remove potential URLs that start with 'www.' without a protocol
    no_www_urls = re.sub(r'\bwww\.\S+\.\S+', '', no_urls)
    
    # Attempt to remove domain references that might not start with 'www.'
    # This is more aggressive and might remove non-URL text that resembles domain patterns
    no_domains = re.sub(r'\b\S+\.(com|org|net|edu|gov|io|co)\b', '', no_www_urls)
    
    return no_domains


def clean_research_text(text):
    """
    Cleans the provided research paper text by removing URLs, references,
    normalizing spaces, and making all text lowercase.
    
    Args:
    text (str): The research paper text to be cleaned.
    
    Returns:
    str: The cleaned text.
    """
    # Remove URLs
    cleaned_text = re.sub(r'https?:\/\/\S+', '', text)
    
    cleaned_text = cleaned_text.replace('...', '.')

    # Remove DOIs
    cleaned_text = re.sub(r'https?://doi.org/[^\s]+', '', cleaned_text)
    
    # Remove References (assuming format [number] or similar markers)
    cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text)
    
    # Normalize text: Removing extra spaces, newlines, and making all text lowercase
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # # Remove any standalone numbers that may have been part of references or enumeration
    # cleaned_text = re.sub(r'\b\d+\b', '', cleaned_text)
    
    # Remove specific headers or footers if they were identified (example placeholder)
    # This step requires customization based on the known structure of the documents
    # cleaned_text = cleaned_text.replace("specific header or footer", "")
    
    return cleaned_text
    
def remove_brackets(text):
    pattern = r'\[.*?\]'
    result = re.sub(pattern, '', text)
    return result

def tokenize_and_count(text):
    sent = sent_tokenize(text)
    for s in sent:
        wrd=word_tokenize(s)
        if(len(wrd)<=5):
            sent.remove(s)
    final_txt= ' '.join(sent)
    return final_txt

def preprocess(text):
    text = remove_brackets(text)
    text = remove_websites(text)
    text = clean_research_text(text)
    text = tokenize_and_count(text)
    return text

import os

def preprocess(text):
    # Your preprocessing steps here
    # For example:
    # Convert text to lowercase
    # Remove special characters
    # Tokenize text
    # Perform other text normalization tasks
    return text

def preprocess_and_save_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        text = input_file.read()
        preprocessed_text = preprocess(text)
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(preprocessed_text)

def preprocess_files_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        preprocess_and_save_file(input_file_path, output_file_path)

# Specify input and output folders
# input_folder = os.path.join("C:\Users","vinay\OneDrive\Desktop\New folder\chemistry")
# output_folder = os.path.join("C:\Users","vinay\OneDrive\Desktop\pre\chemistry")

# preprocess_files_in_folder(input_folder, output_folder)
