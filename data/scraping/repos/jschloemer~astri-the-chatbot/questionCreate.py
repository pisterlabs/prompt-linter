# Designed to be a standalone processor of project text. The goals as follows:
# - 1) Generate questions to be added for NLU processing
# Author: Jeff Schloemer
# Date: 01/25/2023

# pip install openai requests bs4 PyPDF2  nltk textstat textblob

import argparse
import requests
import PyPDF2
import openai
import re
import nltk
from nltk.corpus import cmudict
from bs4 import BeautifulSoup
import os
from textblob import TextBlob
import textstat

key = os.getenv("OPENAI_API_KEY")

# Debug settings
debug = False
useopenai = True

# Globals
max_tokens = 800

# Download the tokenizers and stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def flesch_reading_score(text):
    """Calculate the Flesch Reading Score for a text"""
    return textstat.flesch_reading_ease(text)

# Function for cleaning text
def clean_text(text):
    # Use a regular expression to remove whitespace
    i_text = re.sub(r'\s+', ' ', text)
    c_text = re.sub(r"[^\s\w!@#$5\^&*();:,./?\\<>{}\[\]\-\+=_`~\’\'\"]", ".", i_text)
    return c_text

# Function for filtering sentences
def filter_sentences(sentences, min_length=3, max_length=25):
    # Initialize an empty list to store the filtered sentences
    n_sentences = []

    #iterate through the sentences and check if they meet the criteria
    for sentence in sentences:
        # Length criteria
        le = len(sentence.split())
        if le > min_length and le < max_length:
            n_sentences.append(sentence)
        
            
    s_sentences = []

    #iterate through the sentences and check if they contain stop words
    for sentence in n_sentences:
        words = set(sentence.split())
        if len(words.intersection(stop_words)) < len(words):
            s_sentences.append(sentence)

    # Return the filtered sentences
    return s_sentences

# Function for cleaning sentences
def clean_sentences(sentences):
    # Initialize an empty list to store the clean sentences
    c_sentences = []
    
    #iterate through the sentences and remove extra whitespace
    for sentence in sentences:
        # Length criteria
        w_text = re.sub(r'\s', ' ', sentence)
        p_text = w_text.rstrip("!.?;:")
        text = p_text.strip()
        c_sentences.append(text)
        
    return c_sentences

def find_interesting(sentences):
    para = ""
    para_exemp = ""
    paras = []
    token_count = 0
    
    # Group text into paragraphs
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        if (len(tokens) + token_count) < max_tokens:
            para = para + " " + sentence
            token_count = token_count + len(tokens)
        elif len(tokens) > max_tokens:
            # Check for very long sentences
            continue
        else:
            paras.append(para.strip())
            para = sentence
            token_count = len(tokens)
    
    paras.append(para.strip())
    
    max_score = 0        
    for opt in paras:
        #print(opt)
        #print("==")
        tokens = nltk.word_tokenize(opt)
        flesch = flesch_reading_score(opt)
        text = TextBlob(opt)
        subj = text.sentiment.subjectivity
        polr = text.sentiment.polarity
        
        # Goal is to find harder to read, long, objective and neutral text
        score = (len(tokens) / max_tokens) + ((100 - flesch) / 100) + (1 - subj) + (1 - abs(polr))
        
        if score > max_score:
            para_exemp = opt
            max_score = score  
    
    return para_exemp

def question_sentence(para):
    airesponse = "Default Response"
    key = os.getenv("OPENAI_API_KEY")
    openai.api_key = key
    
    # Use openai to generate a list of questions
    if (useopenai):
        init="Generate 10 questions that someone reading the following text would be able to answer:\n"
        prompt=para
        fin = "\n\n###\n\n"
        total = init + prompt + fin
        response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=total,
                    temperature=0,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=[fin]
                )
        if debug: print(response)
        airesponse = response['choices'][0]['text']
        # Open a text file to write the data to
        with open('question_list.txt', 'w') as txt_file:
            txt_file.write(airesponse)
    
    return airesponse

# Define the command line arguments
parser = argparse.ArgumentParser(description='Text corpus split into sentences')
parser.add_argument('input', help='a text file, PDF or a URL')

# Parse the command-line arguments
args = parser.parse_args()

# Initialize an empty string to store the text
text = ""

# Check if the input is a file, PDF or a URL
if args.input.startswith('http'):
    # Read the text from the URL
    response = requests.get(args.input)
    html = response.text
    
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract the text content
    text = soup.get_text()
elif args.input.endswith('pdf'):
    # Read the text from a PDF
    # Open the PDF file
    with open(args.input, 'rb') as f:
        # Create a PDF object
        pdf = PyPDF2.PdfReader(f)
        
        # Get the number of pages
        num_pages = len(pdf.pages)
        
        # Iterate through each page
        for i in range(num_pages):
            # Extract the text from the page
            page = pdf.pages[i]
            text += page.extract_text()
else:
    # Read the text from the file
    with open(args.input, 'r') as f:
        text = f.read()
        
# Split the text into sentences cleaning the text first
c_text = clean_text(text)

# Tokenize the text
sentences = nltk.sent_tokenize(c_text)
if debug: print(sentences)
if debug: print("================================================")

# Filter out sentences that don't meet criteria
f_sentences = filter_sentences(sentences)
if debug: print(f_sentences)
if debug: print("================================================")

# Clean the specific sentences
c_sentences = clean_sentences(f_sentences)
if debug: print(c_sentences)
if debug: print("================================================")

# Find the most interesting group of sentences
int_sentences = find_interesting(c_sentences)
if debug: print(int_sentences)
if debug: print("================================================")

# Use OpenAI to generated content questions
questions = question_sentence(int_sentences)
if debug: print(questions)
if debug: print("================================================")