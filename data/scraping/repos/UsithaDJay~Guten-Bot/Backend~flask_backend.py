# # !pip install chromadb --progress-bar off
# !pip install replicate
# # !pip install faiss-cpu
# !pip install faiss-gpu
# !pip install transformers --progress-bar off
# !pip install langchain --progress-bar off
# !pip install sentence_transformers --progress-bar off
# !pip install InstructorEmbedding --progress-bar off
# !pip install textsum
# !pip install flask-ngrok
# !pip install pyngrok

# !ngrok authtoken '2XVaUQ29PRt48iMYXxyN6tawIFh_6kZiMroQWZJf812oC2fnz'

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

import os
import glob
import textwrap
import time

import re

import langchain

# loaders
from langchain.document_loaders import TextLoader


# splits
from langchain.text_splitter import RecursiveCharacterTextSplitter

# prompts
from langchain import PromptTemplate, LLMChain

# vector stores
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma

# models
from langchain.llms import HuggingFacePipeline
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import Replicate
from textsum.summarize import Summarizer

# retrievers
from langchain.chains import RetrievalQA

import torch
import transformers
from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from transformers import AutoTokenizer, TextStreamer, pipeline

# data collectors
import requests
from bs4 import BeautifulSoup
import difflib


app = Flask(__name__)
run_with_ngrok(app)


# Models

## Summarizing Model
model_name = "pszemraj/led-large-book-summary"
summarizer = Summarizer(
    model_name_or_path=model_name,
    token_batch_length=10000,
)
# configurations for summarizer
min_word_count = 200
max_word_count = 300

tokens_per_word = 1.3

min_token_count = min_word_count * tokens_per_word
max_token_count = max_word_count * tokens_per_word

# Set the length constraints in the inference params
inference_params = summarizer.inference_params
inference_params['max_length'] = int(max_token_count)
inference_params['min_length'] = int(min_token_count)
summarizer.set_inference_params(inference_params)

summ = pipeline(
    "summarization",
    model_name,
    device=0 if torch.cuda.is_available() else -1,
)

## Embeddings model
instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name = "hkunlp/instructor-base",
        model_kwargs = {"device": "cuda"}
)

## Llama2-13 by Replicate
REPLICATE_API_TOKEN = "r8_4o6DI4Kl9VfQdrVv6OlaqvAyMhFdamr2jUDVe"
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

llm = Replicate(
    model = "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
    input = {"temperature": 0.75, "max_length": 1024, "top_p": 0.95, "repetition_penalty": 1.15},
)


prompt_template = """
Don't try to make up an answer, if you don't know just say that you don't know.
Answer in the same language the question was asked.
Use only the following pieces of context to answer the question at the end.

{context}

Question: {question}
Answer:"""

# Custom Prompt
PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables = ["context", "question"]
)

# Functions for Book Retrieval

## Function to search for a book by name and return the best match URL
def search_book_by_name(book_name):
    base_url = "https://www.gutenberg.org/"
    search_url = base_url + "ebooks/search/?query=" + book_name.replace(" ", "+") + "&submit_search=Go%21"

    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the best match link based on similarity ratio
    best_match_ratio = 0
    best_match_url = ""

    for link in soup.find_all("li", class_="booklink"):
        link_title = link.find("span", class_="title").get_text()
        similarity_ratio = difflib.SequenceMatcher(None, book_name.lower(), link_title.lower()).ratio()
        if similarity_ratio > best_match_ratio:
            best_match_ratio = similarity_ratio
            best_match_url = base_url + link.find("a").get("href")

    return best_match_url

## Function to get the "Plain Text UTF-8" download link from the book page
def get_plain_text_link(book_url):
    response = requests.get(book_url)
    soup = BeautifulSoup(response.content, "html.parser")

    plain_text_link = ""

    for row in soup.find_all("tr"):
        format_cell = row.find("td", class_="unpadded icon_save")
        if format_cell and "Plain Text UTF-8" in format_cell.get_text():
            plain_text_link = format_cell.find("a").get("href")
            break

    return plain_text_link


## Function to get the content of the "Plain Text UTF-8" link
def get_plain_text_content(plain_text_link):
    response = requests.get(plain_text_link)
    content = response.text
    return content


## Main function
def load_book(book_name):
    best_match_url = search_book_by_name(book_name)

    if best_match_url:
        plain_text_link = get_plain_text_link(best_match_url)
        if plain_text_link:
            full_plain_text_link = "https://www.gutenberg.org" + plain_text_link
            plain_text_content = get_plain_text_content(full_plain_text_link)
#             print("Plain Text UTF-8 content:", plain_text_content)

            book_text = plain_text_content

            # Remove the BOM character if it exists
            book_text = book_text.lstrip('\ufeff')

            #####
             # Define the possible variations of the start marker
            possible_start_markers = [
                r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK (.+?) \*\*\*",
                r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK (.+?) \*\*\*"
            ]

            # Fetch the plain_text_content of the book (assuming you have it)
            plain_text_content = book_text  # Fetch the content here

            start_index = None
            for start_marker_pattern in possible_start_markers:
                match = re.search(start_marker_pattern, book_text)
                if match:
                    start_index = match.start()
                    book_name = match.group(1)
                    break

            if start_index is not None:
                end_marker = f"*** END OF THE PROJECT GUTENBERG EBOOK {book_name} ***"

                end_index = plain_text_content.find(end_marker, start_index)

                if end_index != -1:
                    book_text = plain_text_content[start_index + len(match.group(0)):end_index]


            #####

            # Choose an appropriate encoding, such as 'utf-8'
            with open("book.txt", "w", encoding="utf-8") as book:
                book.write(book_text)

            return book_text
        else:
            print("No Plain Text UTF-8 link found.")
            return "web site error"
    else:
        print("No matching book found.")
        return "web site error"


# Function to get Summary
def generate_summary(book_text):
  global summarizer, summ
  out_str = summarizer.summarize_string(book_text)
  wall_of_text = out_str

  result = summ(
      wall_of_text,
      min_length=200,
      max_length=300,
      no_repeat_ngram_size=3,
      encoder_no_repeat_ngram_size=3,
      repetition_penalty=3.5,
      num_beams=4,
      early_stopping=True,
  )
  original_text = result[0]['summary_text']

  # Remove the last sentence
  sentences = original_text.split('. ')
  if len(sentences) > 1:
      final_text = '. '.join(sentences[:-1])
  else:
      final_text = original_text

  # Print the modified text
  print(final_text)

  return final_text


# Functions for Q/A chatbot

## Splitting book.txt to create embeddings
def loadForEmbeddings(txt_file):
    # load document
    loader = TextLoader(txt_file, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 0
    )

    texts = text_splitter.split_documents(documents)
    return texts

def wrap_text_preserve_newlines(text, width=200): # 110
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

## Format llm response
def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])

    sources_used = llm_response['source_documents'][0].metadata['source']

    ans = ans + '\n\nSources: \n' + sources_used
    return ans

## Main function in Q/A
def llm_ans(query):
    start = time.time()
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
        retriever = retriever,
        chain_type_kwargs = {"prompt": PROMPT},
        return_source_documents = True,
        verbose = False
    )
    llm_response = qa_chain(query)
    ans = process_llm_response(llm_response)
    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str

# Example for creating Embeddings
book_name = "The prince"
book_text = load_book(book_name)
book = "book.txt"
texts = loadForEmbeddings(book)

## create embeddings
vectordb = FAISS.from_documents(
    documents = texts,
    embedding = instructor_embeddings
)

# Variable to check whether the book name entered
no_book = False

# Loads book then creates embeddings
@app.route('/submit', methods=['POST'])
def submit():
    global vectordb, retriever, instructor_embeddings, no_book, book_text

    book_name = request.json.get('book_name')
    if not book_name:
      no_book = True
      return jsonify({'status': "Please enter the name of the book."})

    book_text = load_book(book_name)
    if book_text == "web site error":
      return jsonify({'status': 'web site errorr'})
    book = "book.txt"
    texts = loadForEmbeddings(book)

    # create embeddings
    vectordb = FAISS.from_documents(
        documents = texts,
        embedding = instructor_embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs = {"k": 3, "search_type" : "similarity"})

    return jsonify({'status': 'success'})


# generates and returns summary
@app.route('/get_summary', methods=['GET'])
def get_summary():
    global book_text, no_book
    if no_book:
        return jsonify({'answer': "Please enter the name of the book."})
    summary = generate_summary(book_text)

    return jsonify({'book_summary': summary})

# Gets the prompt and returns Llm response
@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.json.get('query')
    # print("QQ:", query)
    if (no_book and not query):
        return jsonify({'answer': "Please enter the name of the book and the prompt."})
    if no_book:
        return jsonify({'answer': "Please enter the name of the book."})
    if not query:
        return jsonify({'answer': "Please enter the prompt."})
    answer = llm_ans(query)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run()