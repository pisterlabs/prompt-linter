import argparse
import torch
import re
import fitz
import arxiv
import os
import openai 

from tqdm import tqdm
from transformers import LlamaTokenizer 
from langchain.text_splitter import TokenTextSplitter


# Init OpenAI, Text_splitter
openai.api_key = os.environ["OPEN_AI_CODEGEN"]
MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
text_splitter = TokenTextSplitter.from_huggingface_tokenizer(tokenizer=tokenizer,
                                                                            chunk_size = 15000,
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
    
    return title, docs, abstract

# OpenAI
SYSTEM_SUMMARY_PROMPT = """
You are a research assistant that summarizes research papers. 
You provide a high level overview of the paper as well as a specific summary of the experiments ran in the paper.
Do not hallucinate or make up any information not present in the paper."
"""

EXPERIMENT_INSTRUCTION_PROMPT = """
Given the content of the research paper, list down in a clear and linear bullet-point format every step of the experiment(s) described. 
Ensure the steps are detailed enough to allow someone to recreate the experiment and eventually code it.
"""

def generate_completion(prompt, system_prompt, messages = [], max_tokens = 1000, model="gpt-3.5-turbo-16k", temperature=0.0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages,
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content

def summary_template(running_summary, doc):
    template = f"""
    Provide a clear and concise summary of the following chunk of a research paper. 
    Do not hallucinate or make up any information not present in the paper.
    Only repsond with bullet points.
    If applicable, a running summary of previous sections will be provided.

    Running Summary:
    {running_summary}
    Given Paper:
    {doc} 
    """
    return template

def experiment_template(abstract, doc):
    prompt = f"""
    Based on the following research paper section,
    please identify and list down the experimental steps:

    Abstract of paper:
    {abstract}
    Given Paper:
    {doc}
    """
    return prompt

# Summarize the paper
def process_paper(docs, abstract):
    """Pass chunks of text through OpenAI to summarize the paper."""

    running_summary = []
    experiments = []

    for doc in docs:
        # Summarize the paper
        sum_prompt = summary_template(running_summary=running_summary, doc=doc)
        summary = generate_completion(sum_prompt, system_prompt=SYSTEM_SUMMARY_PROMPT)
        running_summary.append(summary)
        # Save Experiments
        exp_prompt = experiment_template(abstract=abstract, doc=doc)
        experiment_steps = generate_completion(exp_prompt, system_prompt=EXPERIMENT_INSTRUCTION_PROMPT)
        experiments.append(experiment_steps)

    return running_summary, experiments



# Main function to process a URL
def main():
    parser = argparse.ArgumentParser(description='Summarize Arxiv Paper.')
    parser.add_argument('url', nargs='?', help='Arxiv URL')  # 'nargs=?' makes the URL argument optional
    args = parser.parse_args()

    # If no URL is provided as a command-line argument, ask for one
    if args.url is None:
        args.url = input("Please enter a Arxiv URL: ")
    
    title, docs, abstract = scrape_arxiv(args.url)
    summary, experiments = process_paper(docs=docs, abstract=abstract)

    print(title)
    print(summary)
    print(experiments)

# Run the main function
if __name__ == "__main__":
    main()
