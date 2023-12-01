import re
import time
import requests
import multiprocessing as mp
from ast import literal_eval
from itertools import repeat

import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Comment

import openai
import tiktoken
import backoff 


def google_patents_api_call(patent_id, _restarts = 0, _original_patent_id = ""):
    """
    Makes a call to the Google Patents API.

    If the call is successful, returns the response. If the call is unsuccessful, tries to fix URL and tries again. If the call is still unsuccessful, returns the bad response.
    
    Parameters:
    -----------
    patent_id : str
        Patent ID.
    
    _restarts : int
        Number of times the function has been restarted. Used to prevent infinite recursion.

    Returns:
    --------
    response : requests.models.Response
        Response from Google Patents API.
    """
    if _restarts == 0:
        _original_patent_id = patent_id

    url = f'https://patents.google.com/patent/{patent_id}/en'
    response = requests.get(url)
    if response.ok:
        return response
    else:
        if response.status_code == 404:
            if patent_id[:2] == "US":
                if _restarts == 0:
                    # insert a zero after the first 4 digits after US and try again
                    # for some reason many of the US patents from the pubchem list are mising a zero in the month digit
                    # So this code changes it from "USyyyym..." to "USyyyy0m..."
                    patent_id = _original_patent_id[:6] + "0" + _original_patent_id[6:]
                    return google_patents_api_call(patent_id, _restarts = 1, _original_patent_id = _original_patent_id)
                elif _restarts == 1:
                    # very few patents (~1/1k?) will need this correction. Some are missing the final 1 digit after the letter
                    patent_id = _original_patent_id + "1"
                    return google_patents_api_call(patent_id, _restarts = 2, _original_patent_id = _original_patent_id)
            elif patent_id[:2] == "WO":
                # some WO patents are missing the first two digits of the year. This code tries 19yy, then 20yy if 19yy fails
                if _restarts == 0:
                    # eg. WO9932450A1 -> WO199932450A1
                    patent_id = _original_patent_id[:2] + "19" + _original_patent_id[2:]
                    return google_patents_api_call(patent_id, _restarts = 1, _original_patent_id = _original_patent_id)
                elif _restarts == 1:
                    # eg. WO9932450A1 -> WO1999032450A1
                    patent_id = _original_patent_id[:2] + "19" + _original_patent_id[2:4] + "0" + _original_patent_id[4:]
                    return google_patents_api_call(patent_id, _restarts = 2, _original_patent_id = _original_patent_id)
                elif _restarts == 2:
                    # eg. WO0051639A2 -> WO200051639A2
                    patent_id = _original_patent_id[:2] + "20" + _original_patent_id[2:]
                    return google_patents_api_call(patent_id, _restarts = 3, _original_patent_id = _original_patent_id)
                elif _restarts == 3:
                    # eg. WO9932450A1 -> WO1999032450A1
                    patent_id = _original_patent_id[:2] + "20" + _original_patent_id[2:4] + "0" + _original_patent_id[4:]
                    return google_patents_api_call(patent_id, _restarts = 4, _original_patent_id = _original_patent_id)
    print(f"{response.status_code} ERROR for original ID: {_original_patent_id}. Last tested ID: {patent_id}")
    return response


def remove_non_utf(text):
    cleaned_text = ''
    for char in text:
        try:
            char.encode('utf-8')
            cleaned_text += char
        except UnicodeEncodeError:
            continue
    return cleaned_text


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def fix_text(text):
    # encode greek letters to latin and back to utf-8
    text = text.encode('latin-1', 'ignore').decode('utf-8', 'ignore')

    pattern = re.compile(r'[a-zA-Z0-9\s\(\)\[\]{}\.,;\:\'\-\\\/\+\*\^=\?~`!@#$%&|<>\u0391-\u03C9]+')
    text = ' '.join(pattern.findall(text))
    text = re.sub(r"(\s\w\s\w\s)+", " ", text)
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    text = ' '.join(text.split())

    # remove first occurance of "Abstract" and "Description"
    text = re.sub(r"Abstract", "", text, count=1)
    text = re.sub(r"Description", "", text, count=1)
    # text = unidecode(text)


    # remove patterns of symbols spaced by a space, following space char space char space ...
    pattern = re.compile(r'(\s[\(\)\[\]{}\.,;\:\'\-\\\/\+\*\^=\?~`!@#$%&|<>]+\s)+')
    text = pattern.sub(' ', text)

    # remove patterns like ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) (alky)) ) ) ) ) ) ) ) ) ) ) )
    pattern = re.compile(r'(\)\s)+')
    text = pattern.sub(') ', text)

    return text


def get_patent_info(patent_ids):
    """
    Gets patent abstracts and descriptions for each patent in a set of patent IDs.

    Scrapes the patent abstract and description from Google Patents.
    
    Parameters:
    -----------
    patent_ids : set
        Set of patent IDs.


    Returns:
    --------
    patent_info : dict
        Dictionary of patent IDs to patent abstracts and descriptions. Format is {patent_id: {"abstract": abstract, "description": description}}.
    """
    
    patent_info = dict()
    if len(patent_ids) == 0:
        return patent_info
    for patent_id in patent_ids:
        try:
            # remove dashes from patent_id
            response = google_patents_api_call(patent_id = patent_id.replace("-", ""))
            if response.ok:
                # Scrape patent abstract and description
                html_content = response.text
                soup = BeautifulSoup(html_content, 'html.parser')
                abstract = soup.find('div', class_='abstract')
                description = soup.find('section', {'itemprop': 'description'})
                title = soup.head.find('title')

                if abstract is None:
                    abstract = "None"
                else:
                    abstract = fix_text(abstract.text)

                if description is None:
                    description = "None"
                else:
                    description = fix_text(description.text)

                if title is None:
                    title = "None"
                else:
                    title = fix_text(title.text.split(' - ')[1])
                
                patent_info.update({patent_id: {"abstract": abstract, "description": description, 'title': title}})
        except Exception as e:
            print(patent_id, e)
    return patent_info

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def chatgpt(api_key, model = "gpt-3.5-turbo", system_prompt = "", user_prompt = "", gpt_temperature=0):
    """
    Uses OpenAI GPT Model to generate text.
    
    Parameters:
    -----------
    api_key : str
        OpenAI API key.
    
    model : str
        OpenAI model to use.
    
    system_prompt : str
        System prompt for OpenAI model.
    
    user_prompt : str
        User prompt for OpenAI model.

    gpt_temperature : float
        Temperature for OpenAI model.
    
    Returns:
    --------
    return : str
        Generated text.
    """
    if (user_prompt == "") or (system_prompt == "") or (api_key == "") or (model == ""):
        print("MISSING PROMPT OR API KEY OR MODEL")
        return ""
    try:
        openai.api_key = api_key
        response = completions_with_backoff(
            model=model,
            temperature = gpt_temperature,
            messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return "API REQUEST ERROR"

def summarization_wrapper(patent_info, api_key, model = "gpt-3.5-turbo", system_prompt = "", user_prompt = "", desc_len = 1000, gpt_temperature=0):       
    """
    Wrapper function for chatgpt_summarization.

    Summarizes patent abstract and description for each patent associated with a given molecule. Uses OpenAI GPT Model.

    Parameters:
    -----------
    patent_info : dict
        Dictionary of patent IDs to patent abstracts and descriptions. Format is {patent_id: {"abstract": abstract, "description": description}}.
    
    api_key : str   
        OpenAI API key.

    model : str
        OpenAI model name.

    system_prompt : str
        System prompt for OpenAI model.

    user_prompt : str
        User prompt for OpenAI model.

    desc_len : int
        Maximum length of patent description. Used to lower API cost.

    gpt_temperature : float 
        Temperature for OpenAI model.

    Returns:
    --------
    summarizations : dict
        Dictionary of patent IDs to patent summarizations. Format is {patent_id: summarization}.
    """
    
    if len(patent_info) == 0: # abort if nothing to summarize
        return patent_info
    summarizations = dict()
    for key, value in patent_info.items():
        if len(value["description"]) > desc_len:
            user_prompt_complete = f"{user_prompt}\n\nTitle:\n{value['title']}\n\nAbstract:\n{value['abstract']}\n\nDescription:\n{value['description'][:desc_len]}"
        else:
            user_prompt_complete = f"{user_prompt}\n\nTitle:\n{value['title']}\n\nAbstract:\n{value['abstract']}\n\nDescription:\n{value['description']}"

        # # ensure less than 4k tokens. Most are ~1k
        # enc = tiktoken.get_encoding("cl100k_base")
        # tok = enc.encode(user_prompt_complete)
        # user_prompt_complete = enc.decode(tok[:4000])
        
        # print("*********")
        # print(key)
        # print(user_prompt_complete)
        # summ = ""
        
        summ = chatgpt(api_key = api_key,
                        model = model,
                        system_prompt = system_prompt,
                        user_prompt = user_prompt_complete,
                        gpt_temperature = gpt_temperature,)
        # print(summ)
        # print("*********")
        if summ == "API REQUEST ERROR":
            print(f"API REQUEST ERROR FOR {key}")
        summarizations.update({key: summ})
    return summarizations

def summarizations_to_str(summarizations):
    """
    Converts the GPT summarization to a string of a set that removed duplicates.

    Parameters:
    -----------
    summarizations : dict
        Dictionary of patent IDs to patent summarizations. Format is {patent_id: summarization}.

    Returns:
    --------
    str
        String of a set that removed duplicates.
    
    """
    
    if not isinstance(summarizations, dict):
        return str(set())
    if summarizations == {}:
        return str(set())
    s_list = list(summarizations.values())
    # split on ' / '
    s_list = [s.split(' / ') for s in s_list]
    # flatten list
    s_list = [item for sublist in s_list for item in sublist]
    # remove empty strings and 'NA', and "API REQUEST ERROR"
    s_list = [s for s in s_list if s not in ['', 'NA', 'API REQUEST ERROR']]
    return str(set(s_list))


def main():
    t_curr = time.time()
    output_dir = "../results/surechembl_smiles_canon_chiral_randomized_patents_l10p_summarizations"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    head_n = 100
    desc_len = 3500
    gpt_temperature = 0
    gpt_model = "gpt-3.5-turbo"
    # gpt_model = "gpt-4"
    api_key = ""
    gpt_summ_system_prompt = "You are an organic chemist summarizing chemical patents"
    gpt_summ_user_prompt = r"Return a short set of three 1-3 word descriptors that best describe the chemical or pharmacological function(s) of the molecule described by the given patent title, abstract, and partial description (giving more weight to title & abstract). Be specific and concise, but not necessarily comprehensive (choose a small number of great descriptor). Follow the syntax '{descriptor_1} / {descriptor_2} / {etc}', writing 'NA' if nothing is provided. DO NOT BREAK THIS SYNTAX. The following is the patent:"
    df = pd.read_csv('../data/surechembl_smiles_canon_chiral_randomized_patents_l10p_noNA.csv', nrows=head_n)
    df["patent_ids"] = df["patent_ids"].map(literal_eval)
    print(f"Time to read in data: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds.")

    # df = df.iloc[0:100]
    # df = df.iloc[100:200]

    n_cpus = 16
    print(f"INFO: Using {n_cpus} CPUs")
    with mp.Pool(n_cpus) as p:
        patent_info = p.map(get_patent_info, df["patent_ids"].tolist())
        print(f"Time to get patent info: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds.")
        summarization_sources = p.starmap(summarization_wrapper, zip(patent_info, repeat(api_key), repeat(gpt_model), repeat(gpt_summ_system_prompt), repeat(gpt_summ_user_prompt), repeat(desc_len), repeat(gpt_temperature)))
        summarizations = p.map(summarizations_to_str, summarization_sources)
        print(f"Time to get summarizations: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds.")
    df["summarization_sources"] = [s for s in summarization_sources]
    df["summarizations"] = [s for s in summarizations]

    # append number to end of save path to avoid overwriting
    i = 1
    save_path = f"{output_dir}/surechembl_summarizations_top-{head_n}_{gpt_model}_desc-{desc_len}_{i}"
    if Path(f"{save_path}.csv").exists():
        while Path(f"{save_path}.csv").exists():
            save_path = f"{output_dir}/surechembl_summarizations_top-{head_n}_{gpt_model}_desc-{desc_len}_{i}"
            i += 1

    df[["smiles", "cid", "patent_ids", "summarization_sources", "summarizations"]].to_csv(f"{save_path}.csv", index=False)

    # save prompt
    with open(f"{save_path}_prompt.txt", "w") as f:
        f.write(gpt_summ_system_prompt + "\n\n" + gpt_summ_user_prompt)
    print(f"Time to save: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds.")


if __name__ == "__main__":
    main()