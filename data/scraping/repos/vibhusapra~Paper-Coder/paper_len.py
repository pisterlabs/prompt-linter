import argparse
import torch
import re
import fitz
import arxiv
import os

from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer 
from langchain.text_splitter import TokenTextSplitter


# Init Openia, Text_splitter
MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
text_splitter = TokenTextSplitter.from_huggingface_tokenizer(tokenizer=tokenizer,
                                                                            chunk_size = 2500,
                                                                            chunk_overlap = 100,
                                                                            disallowed_special=())

# Scrape Arxiv Papers
def extract_id(url):
    """Extract the paper ID out from the url."""
    doc_id = url.rsplit('/', 1)[-1]
    doc_id = re.match(r"[\d\.]*\d", doc_id)

    return doc_id.group(0) if doc_id else ""

def scrape_arxiv(url):
    """Download and Arxiv paper and return chunked text"""
    url = url.lower()
    paper_id = extract_id(url) # Get paper ID
    search = arxiv.Search(paper_id, max_results=1) # ArXiv API to get paper
    result = next(search.results(), None) 
    title = result.title 
    abstract = result.summary
    doc_file_name = result.download_pdf() # ArXiv API to download paper
    with fitz.open(doc_file_name) as doc_file:
        text = "".join(page.get_text() for page in doc_file)
    os.remove(doc_file_name) # Delete paper
    docs = text_splitter.split_text(text)
    tokens = tokenizer.encode(text, return_tensors="pt")
    paper_length = tokens.shape[1]
    
    return title, docs, abstract, paper_length


# Main function to process a URL
def main():
    parser = argparse.ArgumentParser(description='Summarize Arxiv Paper.')
    parser.add_argument('url', nargs='?', help='Arxiv URL')  # 'nargs=?' makes the URL argument optional
    args = parser.parse_args()

    # If no URL is provided as a command-line argument, ask for one
    if args.url is None:
        args.url = input("Please enter a Arxiv URL: ")

    title, docs, abstract, paper_length = scrape_arxiv(args.url)
    print(f"Title: {title}")
    # print(f"Abstract: {abstract}")
    print(f"Paper Length: {paper_length}")

# Run the main function
if __name__ == "__main__":
    main()