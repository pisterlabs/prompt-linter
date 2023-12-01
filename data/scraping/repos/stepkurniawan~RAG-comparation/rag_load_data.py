from datasets import load_dataset, Dataset
from langchain.document_loaders import WebBaseLoader, HuggingFaceDatasetLoader
from langchain.document_loaders import JSONLoader




import os
import pandas as pd


WIKI_PATH = "data/Sustainability+Methods_dump.xml"
JSON_PATH = "data/Sustainability+Methods_dump.xml.json"
QA_GT_JSON_PATH = "collection_ground_truth_ragas_chatgpt4.json"


def get_arxiv_data_from_dataset():
    """
    REF: https://github.com/pinecone-io/examples/blob/master/learn/generation/llm-field-guide/llama-2/llama-2-13b-retrievalqa.ipynb
    ex query: Explain to me the difference between nuclear fission and fusion.
    """
    data = load_dataset(
        'jamescalam/llama-2-arxiv-papers-chunked',
        split='train'
    )
    print(f'success load data: {data}')
    return data

def get_wikipedia_data_from_dataset():
    # output : DatasetDict (from hugging face)
    data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')
    print(f'success load data: {data}')
    return data

def load_sustainability_wiki_dataset():
    # its in hf_dataset format, just like get_wikipedia_data_from_dataset
    dataset_name = "stepkurniawan/sustainability-methods-wiki"
    dataset_from_hf = load_dataset(dataset_name, split='train')
    print(f'success load data from huggingface: stepkurniawan/sustainability-methods-wiki')

    return dataset_from_hf

def load_sustainability_wiki_langchain_documents():
    # the result is Document(page_content=...) just like load_from_webpage()
    dataset_name = "stepkurniawan/sustainability-methods-wiki"
    page_content_column = "text"

    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
    data = loader.load()
    print(f'success load data from huggingface: stepkurniawan/sustainability-methods-wiki')
    return data



def load_from_webpage(link):
    loader = WebBaseLoader(link)
    data = loader.load()
    print(f'success load data from webpage: {link}')
    return data

def get_qa_dataframe():
    """
    get question answer dataframe 
    REF: RAGAS quickstart https://colab.research.google.com/github/explodinggradients/ragas/blob/main/docs/quickstart.ipynb#scrollTo=f17bcf9d
    the dataset must consists of: 
    1. question
    2. ((answer)) -> this will be populated later, when the RAG model answers the question
    3. contexts
    4. ground_truths
    but in this case, we only consider 1 context and 1 ground_truth
    """

    loader = JSONLoader(
    file_path=QA_GT_JSON_PATH,
    jq_schema='.messages[].content',
    text_content=False)


    qa_gt_df = loader.load() # type: Documents

    return qa_gt_df
