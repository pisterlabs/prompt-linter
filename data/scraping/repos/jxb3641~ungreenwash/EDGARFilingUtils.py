"""Utilities for 10K filings."""
import streamlit as st
import glob
import re
import random
import json
import pandas as pd
import nltk
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
from pathlib import Path


import openai
openai.api_key = st.secrets["openai_api_key"]


TICKER_TO_COMPANY_NAME = {
    "F": "Ford Motor Company",
    "GIS": "General Mills, Inc.",
    "PEP": "PepsiCo, Inc.",
    "FSR": "Fisker, Inc."
}
QUESTION_TO_CATEGORY = {
"What does this company do?": "General", 
"What are the risks this company faces?": "General",
"What are the environmental risks this company faces?": "General",
"What are the climate-related risks and opportunities the organization has identified over the short, medium and long term?":"General",
"Environmental laws, environmental risks, environmental regulations": "General"
}
ROOT_DATA_DIR = Path("data/ind_lists")

#TODO: Refactor this into two functions: 
# one that takes in a submission id, and creates a dict of item1/mda sources, and texts
# another that takes in the directory, and outputs all submission ids
# random sample filings should be done separately, or in a third function.

def get_all_submission_ids(datadir=ROOT_DATA_DIR/'4_food_bev'/'10k'):
    """get all the submission IDs of 10-K .txts.  
    Assumes filing texts are of form (submission-id).txt, (submission-id)_item1.txt, (submission-id)_mda.txt

    Args:
        datadir (str): Where to look for the text files.
    Returns:
        (tuple(str)): Tuple of unique submission IDs.
    """

    tenk_all_filingnames = sorted(set([re.search("([A-Z]+)_\d+-\d+-\d+",str(fp)).group() for fp in datadir.glob("*.txt")]))

    return tuple(tenk_all_filingnames)

def get_text_from_files_for_submission_id(filename, datadir=ROOT_DATA_DIR/'4_food_bev'/'10k'):
    """Read in the .txt files for submission_id, located in datadir. 

    Args:
        filename (str): Submission id of the filing.
        datadir (str): filepath where all 3 files (.txt, item1.txt, mda.txt) for the submission id should be located.  

    Returns:
        dict: Dictionary containing the submission id, filepath of the .txt, item1.txt and mda.txt, files, 
        and their texts read in as strings with keys full_txt, item1_txt, mda_txt.
    """

    

    text_dict = {}
    for fp in datadir.glob(f"{filename}*.txt"):
        if re.search("item1.txt",str(fp)):
            text_dict["item1"] = str(fp)
            text_dict["item1_txt"] = fp.read_text(encoding="utf-8").replace("$","\$")
        elif re.search("mda.txt",str(fp)):
            text_dict["mda"] = str(fp)
            text_dict["mda_txt"] = fp.read_text(encoding="utf-8").replace("$","\$")
        else:
            text_dict["fullFiling"] = str(fp)
            text_dict["full_txt"] = fp.read_text(encoding="utf-8").replace("$","\$")

    return text_dict


def get_random_sample_filings(number_filings=50,seed=None):
    """For a random sample of filings, parse their names, MDA, and Item1 .txt files
    and their text.

    Args:
        seed (int, optional): Seed for random instance. Defaults to None.
        number_filings (int, optional): Number of filings to get the MDA, Item1 and full .txt files. Defaults to 50.

    Returns:
        pandas.DataFrame: DF of filing names, the filepaths of the Full, MDA and Item1 text, and their parsed in text. 
    """

    # Helper function to read in file texts as strings
    def get_text(fp):
        with open(fp) as f:
            text = f.read()
        return text

    random_inst = random.Random(seed) if seed else random.Random()

    # All .txt files in the data directory have name {digits}-{digits}-{digits} as their prefix, and 
    # one of _mda.txt, _item1.txt, or just .txt as suffixes. The RE below just captures the common prefixes.
    tenk_all_filingnames = [re.search("\d+-\d+-\d+",fp).group() for fp in glob.glob("data/10K/q1/*.txt")]

    # Pull number_filings (fullFiling, MDA, and Item1) filename triples
    txt_by_filing = {}
    for filing_num in random_inst.sample(tenk_all_filingnames,number_filings):
        txt_by_filing[filing_num] = {}
        for fp in glob.glob(f"data/10K/q1/{filing_num}*.txt"): # Find the 3 files with the common filing prefix
            if re.search("item1.txt",fp):
                txt_by_filing[filing_num]["item1"] = fp
            elif re.search("mda.txt",fp):
                txt_by_filing[filing_num]["mda"] = fp
            else:
                txt_by_filing[filing_num]["fullFiling"] = fp

    # DF indexed by filing prefix, with columns "item1", "mda", "fullFiling".  
    # Add in 3 more columns to contain the text strings.
    df = pd.read_json(json.dumps(txt_by_filing),orient='index')
    for col in df.columns:
        df[col+"_txt"] = df[col].apply(lambda x: get_text(x))
    return df

def split_text(text,form_type=None):
    """split text into workable chunks. Filter out header and footer.

    Args:
        text (str): original text.
        form_type (str, optional): flag to customize splitting for different form types. Implements rolling text chunking with 5 sentences for 10KItemsOnly texts. 
        Default None.

    Returns:
        list(str): list of text chunks.
    """

    if form_type == "10KItemsOnly":
        # Implement super basic chunking (as big as possible, with a sliding window of 3 sentences)
        split_text = []
        sentences = nltk.sent_tokenize(text.replace(";",".").replace("\u2022",""))
        chunk = ""
        chunk_index = 0
        previous_sentences_token_len = 0
        for sent_ind, sentence in enumerate(sentences):
            #Collect chunks with up to 1800 tokens. 
            if len(tokenizer.encode(chunk)) <= 700-previous_sentences_token_len:
                chunk += f" {sentence}"
            else: #Chunk token limit reached.  
                chunk = chunk.strip() #Get rid of leading/trailing whitespace
                chunk_index += 1
                if chunk_index %10 == 0:
                    print(chunk_index,"chunks processed.")
                if chunk_index > 1: #For any chunks after the first
                    # Add in up to last N sentences of last chunk to this one, making sure we dont wrap around to negative indices 
                    if sent_ind -4 >= 0:
                        previous_sentences = " ".join([sentences[sent_ind-4],sentences[sent_ind-3], sentences[sent_ind-2], sentences[sent_ind-1]]) 
                    elif sent_ind -3 >= 0:
                        previous_sentences= " ".join([sentences[sent_ind-3], sentences[sent_ind-2], sentences[sent_ind-1]]) 
                    elif sent_ind -2 >= 0:
                        previous_sentences= " ".join([sentences[sent_ind-2], sentences[sent_ind-1]]) 
                    elif sent_ind -1 >= 0:
                        previous_sentences= " ".join([sentences[sent_ind-1]]) 
                    else:
                        previous_sentences = ""
                    previous_sentences_token_len = len(tokenizer.encode(previous_sentences))
                    #print("\n\nBEFORE:\n\n", chunk)
                    #print("\n\n LENGTH:",len(tokenizer.encode(chunk)))
                    #print()
                    chunk = " ".join([previous_sentences,chunk])
                    #print(f"AFTER INCORPORATING SENTENCES BEFORE {sent_ind}:\n\n", chunk)
                    #print("\n\n LENGTH:",len(tokenizer.encode(chunk)))
                    #print()
                if len(tokenizer.encode(chunk)) <2048:
                    split_text.append(chunk)
                # Add in the current sentence.
                chunk = sentence
        return split_text


    
    split_text = text.split("\n\n")
    start_index = 0 # Find the "Washington, DC" chunk, we will throw out all other chunks before this 
    end_index = -1 # Find the the "Item 15" chunk, we will throw out all chunks after this
    for i, chunk in enumerate(split_text):
        if re.search("washington,",chunk.lower()):
            start_index = i
#        elif re.search(r"item 15\.",chunk.lower()):
#            end_index = i



    return split_text[start_index+1:end_index] 


def filter_chunks(split_text):
    """Filter split chunks."""

    filtered_split = [] 
    #Remove chunks less than some hard limit in length 
    for chunk in split_text:
        if len(chunk.split())>=15:
            filtered_split.append(chunk)

    return filtered_split 
  

def does_text_have_climate_keywords(text):
    """Checks if any of a preset list of keywords is in the text.

    Args:
        text (str): text to search for keywords.
    Returns:
        A dict of sentences featuring keywords, a dict of keyword counts in the text.
    """

    keywords = [
        "energy",
        "electric vehicle",
        "climate change",
        "wind (power|energy)",
        "greenhouse gas",
        "solar",
        "\bair\b",
        "carbon",
        "emission",
        "extreme weather",
        "carbon dioxide",
        "battery",
        "pollution",
        "environment",
        "clean power",
        "onshore",
        "coastal area",
        "footprint",
        "charge station",
        "eco friendly",
        "sustainability",
        "energy reform",
        "renewable",
    ]

    keyword_contexts = {keyword : [] for keyword in keywords}
    keyword_counts = {keyword : 0 for keyword in keywords}

    # pre-process text
    split_text = text.lower().split(". ")

    # Count occurrences for each keyword in the text.
    for keyword in keywords:
        for sentence in split_text:
            if re.search(keyword,sentence):
                keyword_contexts[keyword].append(sentence)
                keyword_counts[keyword] = keyword_counts[keyword] + len(re.findall(keyword,sentence))
    return keyword_contexts, keyword_counts

def concat_keyword_sentences(keyword_sentence_map,max_str_length=900):
    """Take in a dictionary of keyword to sentences, and concatenate them up to max_str_length.

    Args:
        keyword_sentence_map (dict): dictionary of sentences by keyword.
        max_str_length (int, optional): maximum length of the concated string. Defaults to 900.

    Returns:
        str: concatenated string of keyword sentences, of length approximately max_str_length characters.
    """

    keyword_sentence_list = [ sent for sentlist in keyword_sentence_map.values() for sent in sentlist]
    concat_str = ""
    while len(concat_str)<max_str_length:
        for keyword_sentence in keyword_sentence_list:
            concat_str += keyword_sentence+"\n\n" 
    return concat_str


def get_chunks_from_file(filename):
    import csv
    chunks = []
    with open(filename) as f:
        #skip header
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            if row[3] == "Firm":
                if row[4] and len(row[4]) > 75:
                    if len(row[4]) > 200:
                        sentences = nltk.sent_tokenize(row[4])
                        #create chunks of 8 sentences
                        for i in range(0, len(sentences), 8):
                            chunk = "".join(sentences[i:i+8])
                            if chunk:
                                chunks.append(chunk)
                    else:
                        chunks.append(row[4])
    return chunks

def get_chunks_from_esg_report(filename):
    with open(filename) as f:
        text = f.read()
        chunks = []
        for line in text.split("\n\n"):
            line = line.replace(' \n', ' ').replace('.\n', '.').replace('\r', '')
            print(line)
            if line and len(line) > 50:
                chunks.append(line)
    return chunks


def get_big_chunks_from_esg_report(filename):
    with open(filename) as f:
        text = f.read()
        text = text.replace(' \n', ' ').replace('.\n', '.').replace('\r', '')
        sentences = nltk.sent_tokenize(text)
        #create chunks of 100 sentences
        chunks = []
        for i in range(0, len(sentences), 20):
            chunk = "".join(sentences[i:i+20])
            if chunk:
                chunks.append(chunk)
    return chunks



if __name__ == "__main__":
    from OpenAIUtils import file_to_embeddings, questions_to_answers

    filename = "/Users/colemanhindes/hackathon/OpenAI-hackathon-Scope3/data/ind_lists/4_food_bev/sustainability_reports/gis_2022.txt"

    text_questions = '''What does this company do? 
What are the risks this company faces?
What are the environmental risks this company faces?
"What are the climate-related risks and opportunities the organization has identified over the short, medium, and long term?
What is the impact of climate-related risks and opportunities on the organization’s business, strategy, and financial planning.
What are the organization’s processes for identifying and assessing climate-related risks? 
What are extreme climate events the firm is exposed to?
What are lasting changes in the climate the firm is exposed to?
What are climate-related regulations, rules, bills or standards that the entity must adhere to?
What are new technologies that the entity is considering or requiring to decarbonize its business model?
What are the reputational risks or concerns that the firm attributes to climate- or corporate socical responsibility-related issues?
Does the firm rely on or employ any kind of green financing?
Has the firm set up a committee (or other governance mechanism) that is concerned with climate- and ESG-related issues? 
Has the firm set up a committee (or other governance mechanism) that is concerned with climate- and ESG-related issues? 
What does the company disclose about its energy mix? 
What is the percentage of energy or electricity used that is from renewable sources?
What are the company's emissions targets and have they been validated as credible and substantial? 
Does this company's emissions targets include Scope 3 (indirect) emissions?
Does a discussion of long-term and short-term strategy or plan to manage Scope 1 emissions, emissions reduction targets, and an analysis of performance toward those targets exist? 
What does the company say about its  impacts on biodiversity? 
What does the company disclose about the waste it generates and what is it doing to reduce waste? 
What are key aspects of "Product Safety" that the firm discusses in its sustainability report?
What are key aspects of "Labor Practices" that the firm discusses in its sustainability report?
What are key aspects of "Fuel Economy & Use-phase Emissions" that the firm discusses in its sustainability report?
What are key aspects of "Material Sourcing" that the firm discusses in its sustainability report?
What are key aspects of "Materials Efficiency & Recycling" that the firm discusses in its sustainability report?
What are key aspects of "Water Management" that the firm discusses in its sustainability report?
What are key aspects of "Food Safety" that the firm discusses in its sustainability report?
What are key aspects of "Health & Nutrition" that the firm discusses in its sustainability report?
What are key aspects of "Ingridient Sourcing" that the firm discusses in its sustainability report?
What is this company's strategy to reduce the environmental impact of packaging throughout its lifecycle? 
What is the company doing about the environmental and social impacts of their ingredient supply chain?'''

    questions = text_questions.split("\n")

    #Only show matches above this level
    match_threshold = 0.35

    chunks = get_big_chunks_from_esg_report(filename)
    for chunk in chunks:
        if not chunk:
            print("empty chunk")
    embeddings = file_to_embeddings(Path(filename), chunks)
    answers = questions_to_answers(questions, embeddings, min_similarity=match_threshold)
    for question, answer in zip(questions, answers):
        prompt = f"If the answer to the question is in the excerpt below, answer it or else say N/A\n START CONTEXT\n{answer}\nEND CONTEXT \nAnswer the following question: {question}\nAnswer:"

        response = openai.Completion.create(
            model='text-davinci-002',
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            stop=["\n","."]
        ).choices[0].text
        print(f"Question: {question}")
        print(f"Answer: {response}")



        

