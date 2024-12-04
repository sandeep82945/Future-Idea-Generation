import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import string
import os
import re

nltk.download('punkt')

def remove_brackets(text):
    pattern = r'\[.*?\]'
    result = re.sub(pattern, '', text)
    return result

def tokenize_and_count(folder_path):
    total_tokens = 0
    count=[]
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            f=0
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            ftext=remove_brackets(text)
            sent = sent_tokenize(ftext)
            for s in sent:
                wrd=word_tokenize(s)
                if(len(wrd)<=6):
                    sent.remove(s)
                else:
                    f+=len(wrd)
            count.append(f)
            if(f>20000):
                print(f)
                print(filename)
    count.sort()
    print(len(count))
    print(count[0])
    s=0      
    for i in count:
        s+=i
    return s/len(count)

folder_path=os.path.join("D:\dsa","final\comp22")
num_tokens = tokenize_and_count(folder_path)
print(num_tokens)

