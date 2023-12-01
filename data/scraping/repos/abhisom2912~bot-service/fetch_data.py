
from transformers import GPT2TokenizerFast
import numpy as np
from github import Github
from dotenv import dotenv_values
import time
import pyparsing as pp
import openai
import pandas as pd
import tiktoken
from nltk.tokenize import sent_tokenize
import nltk
import requests
import ssl

from utilities.scrapers.gitbook_scraper import *
from utilities.scrapers.pdf_parse_seq import *
from utilities.scrapers.medium_parser import *

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
config = dotenv_values("../.env")

openai.api_key = config['OPENAI_API_KEY']
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "cl100k_base"
min_token_limit = 10
EMBEDDING_COST = 0.0004
COMPLETIONS_COST = 0.03
max_len = 1500

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}


# functions to fetch data
def find_all(s, ch):
    previous_ind = 0
    array = []
    length = len(s)
    while 1:
        try:
            ind = s.index(ch)
            array.append(ind + previous_ind)
            s = s[ind + len(ch):length]
            previous_ind = previous_ind + ind + len(ch)
        except ValueError:
            break
    return array

def remove_unwanted_char(s):
    code_separator = "```"
    index_array = find_all(s, code_separator)
    i = 0
    if len(index_array) % 2 == 1:
        index_array.append(len(s))
    while i < len(index_array):
        start_index = index_array[i]
        i = i+1
        end_index = index_array[i]
        orig_string = s[start_index:end_index]
        replaced_string = orig_string.replace('#', '--')
        s = s.replace(orig_string, replaced_string)
        i = i+1
    return s

def get_needed_hash(s):
    s_array = s.split("\n")
    i = len(s_array) - 1
    req_no_of_hash = 2
    while i > 0:
        if s_array[i].find("#") != -1:
            req_no_of_hash = s_array[i].count('#') + 1
            break
        i = i - 1
    no_hash = 0
    hash_string = ''
    while no_hash < req_no_of_hash:
        hash_string = hash_string + '#'
        no_hash = no_hash + 1
    return hash_string

def cleanup_data(s):
    s = remove_unwanted_char(s)
    s = s.replace('<details>', '')
    s = s.replace('</details>', '')
    s = s.replace('</b></summary>', '')
    # hash_string = get_needed_hash(s[0:s.find('<summary><b>')])
    hash_string = ''
    s = s.replace('<summary><b>', hash_string)
    return s

def clean_content(content):
    s1 = '<Section'
    s2 = '</Section>'
    remove_content = find_between(content, s1, s2)
    content = content.replace(remove_content,'').replace(s1, '').replace(s2, '')
    return content

def read_docs(github_repo, github_directory):
    g = Github(config['GITHUB_ACCESS_TOKEN'])
    repo = g.get_repo(github_repo)
    title_stack = []
    contents = repo.get_contents("")
    file_content = ''
    while contents:
        try:
            file_content = contents.pop(0)
        except Exception:
            pass
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            if (github_directory != '' and file_content.path.find(github_directory) == -1):  ## remove orchestrator line later
                continue
            if file_content.name.endswith('md') or file_content.name.endswith('mdx'):
                file_contents = repo.get_contents(file_content.path)
                title = pp.AtLineStart(pp.Word("#")) + pp.rest_of_line
                sample = file_contents.decoded_content.decode()
                sample = cleanup_data(sample)

                title_stack.append([0, 'start_of_file']) # level, title, content, path
                if sample.split('\n')[0] == '---':
                    title_stack[-1].append('')
                    title_stack[-1].append(file_content.path.replace(github_directory, ''))
                    title_stack.append([1, sample.split('\n')[1].split(':')[1].lstrip()])
                    sample = sample.split('---')[2]

                last_end = 0
                for t, start, end in title.scan_string(sample):
                    # save content since last title in the last item in title_stack
                    title_stack[-1].append(clean_content(sample[last_end:start].lstrip("\n")))
                    title_stack[-1].append(file_content.path.replace(github_directory, ''))

                    # add a new entry to title_stack
                    marker, title_content = t
                    level = len(marker)
                    title_stack.append([level, title_content.lstrip()])

                    # update last_end to the end of the current match
                    last_end = end

                # add trailing text to the final parsed title
                title_stack[-1].append(clean_content(sample[last_end:]))
                title_stack[-1].append(file_content.path.replace(github_directory, ''))
    return title_stack

def create_data_for_docs(protocol_title, title_stack, doc_link, doc_type):
    heads = {}
    max_level = 0
    nheadings, ncontents, ntitles, nlinks = [], [], [], []
    outputs = []

    for level, header, content, dir in title_stack:
        final_header = header
        dir_header = ''

        if doc_type == 'pdf':
            content_link = doc_link
            title = protocol_title + " - whitepaper"
        elif doc_type == 'gitbook':
            content_link =  dir
            title = protocol_title + " - whitepaper"
            dir_elements = dir.replace('https://', '').split('/')
            element_len = 1
            while element_len < len(dir_elements) - 1:
                dir_header += dir_elements[element_len].replace('-', ' ') + ': '
                element_len += 1
        elif doc_type == 'medium':
            content_link = dir
            title = protocol_title + " - articles"

        else:
            element_len = 1
            dir_elements = dir.split('/')
            content_link = doc_link + '/' + dir_elements[0]
            sub = 1
            title = protocol_title + " - " + dir_elements[0]
            if dir_elements[len(dir_elements) - sub].find('README') != -1:
                sub = sub + 1
            while element_len < len(dir_elements) - sub:
                dir_header = dir_header + dir_elements[element_len] + ': '
                element_len = element_len + 1

            element_len = 1
            while element_len < len(dir_elements) - sub + 1:
                if dir_elements[element_len].find('.md'):
                    link = dir_elements[element_len].replace('.mdx', '').replace('.md', '')
                content_link = content_link + '/' + link
                element_len = element_len + 1

        if level > 0:
            heads[level] = header
            if level > max_level:
                max_level = level
            while max_level > level:
                try:
                    heads.pop(max_level)
                except Exception:
                    pass
                max_level = max_level - 1

        i = level - 1
        while i > 0:
            try:
                final_header = heads[i] + ': ' + final_header
            except Exception:
                pass
            i = i - 1
        final_header = dir_header + final_header
        if final_header.find('start_of_file') == -1:
            if content.strip() == '':
                continue
            nheadings.append(final_header.strip())
            ncontents.append(content)
            ntitles.append(title)
            nlinks.append(content_link)


    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]
    for title, h, c, t, l in zip(ntitles, nheadings, ncontents, ncontent_ntokens, nlinks):
        if (t < max_len and t > min_token_limit):
            outputs += [(title, h, c, t, l)]
        elif (t >= max_len):
            outputs += [(title, h, reduce_long(c, max_len), count_tokens(reduce_long(c, max_len)), l)]
    return outputs

def final_data_for_openai(outputs):
    res = []
    res += outputs
    df = pd.DataFrame(res, columns=["title", "heading", "content", "tokens", "link"])
    df = df[df.tokens>10]   # to ensure really small and insignificant data doesn't get indexed
    df = df.drop_duplicates(['title','heading'])
    df = df.reset_index().drop('index',axis=1) # reset index
    df = df.set_index(["title", "heading"])
    return df


def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""

def reduce_long(
        long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text

def get_embedding(text: str, model: str = EMBEDDING_MODEL):
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"], result["usage"]["total_tokens"]

def compute_doc_embeddings(df: pd.DataFrame):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    embedding_dict = {}
    total_tokens_used = 0
    for idx, r in df.iterrows():
        embedding, tokens = get_embedding(r.content)
        embedding_dict[idx] = embedding
        total_tokens_used = total_tokens_used + tokens
        time.sleep(7)
    cost_incurred = total_tokens_used * EMBEDDING_COST / 1000
    print(cost_incurred)
    return embedding_dict, cost_incurred

def read_from_github(protocol_title, github_link, github_doc_link, github_directory):
    github_repo = github_link.partition("github.com/")[2]
    print(github_repo)
    title_stack = read_docs(github_repo, github_directory)
    outputs = create_data_for_docs(protocol_title, title_stack, github_doc_link, 'github')
    print(outputs)
    df = final_data_for_openai(outputs)
    print(df.head)
    document_embeddings, cost_incurred = compute_doc_embeddings(df)
    print(len(df), " rows in the data.")
    return outputs, document_embeddings, cost_incurred

def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content

def add_data_array(file_path, content):
    title_stack = []
    title = pp.AtLineStart(pp.Word("#")) + pp.rest_of_line
    title_stack.append([0, 'start_of_file'])
    if content.split('\n')[0] == '---':
        title_stack[-1].append('')
        title_stack[-1].append(file_path)
        title_stack.append([1, content.split('\n')[1].split(':')[1].lstrip()])
        content = content.split('---')[2]
    last_end = 0
    for t, start, end in title.scan_string(content):

        # save content since last title in the last item in title_stack
        title_stack[-1].append(content[last_end:start].lstrip("\n"))
        title_stack[-1].append(file_path)

        # add a new entry to title_stack
        marker, title_content = t
        level = len(marker)
        title_stack.append([level, title_content.lstrip()])

        # update last_end to the end of the current match
        last_end = end

    # add trailing text to the final parsed title
    title_stack[-1].append(content[last_end:])
    title_stack[-1].append(file_path)
    return title_stack

# handling request to get data from Gitbook
def get_data_from_gitbook(gitbook_data_type, gitbook_link, protocol_title):
    https_str = "https://"
    if gitbook_link[len(gitbook_link)-1] == "/":
        gitbook_link = gitbook_link[0 : len(gitbook_link)-1]
    inter_str = gitbook_link.replace(https_str, '')
    base_url = https_str + (inter_str.split('/', 1)[0] if len(inter_str.split('/', 1)) > 1  else inter_str)
    first_url = '/' + inter_str.split('/', 1)[1] if len(inter_str.split('/', 1)) > 1  else ''
    title_stack = get_gitbook_data(base_url, first_url, gitbook_data_type)
    # title_stack = get_gitbook_data(gitbook_link, '', gitbook_data_type)
    outputs = create_data_for_docs(protocol_title, title_stack, '', 'gitbook')
    print('Outputs created for gitbook data')
    df = final_data_for_openai(outputs)
    print(df.head)
    document_embeddings, cost_incurred = compute_doc_embeddings(df)
    print('Embeddings created, sending data to db...')
    return outputs, document_embeddings, cost_incurred

# handling request to get data from a PDF document
def get_pdf_whitepaper_data(document, table_of_contents_pages, whitepaper_link, protocol_title):
    content = convert_to_md_format(document, table_of_contents_pages)
    title_stack = add_data_array('whitepaper', content)
    outputs = create_data_for_docs(protocol_title, title_stack, whitepaper_link, 'pdf')
    print('Outputs created for whitepaper data')
    df = final_data_for_openai(outputs)
    print(df.head)
    document_embeddings, cost_incurred = compute_doc_embeddings(df)
    print('Embeddings created, sending data to db...')
    return outputs, document_embeddings, cost_incurred

# handling request to get data from Medium
def get_data_from_medium(username, valid_articles_duration_days, protocol_title):
    title_stack = get_medium_data(username, valid_articles_duration_days)
    outputs = create_data_for_docs(protocol_title, title_stack, '', 'medium')
    print('Outputs created for gitbook data')
    df = final_data_for_openai(outputs)
    print(df.head)
    document_embeddings, cost_incurred = compute_doc_embeddings(df)
    print('Embeddings created, sending data to db...')
    return outputs, document_embeddings, cost_incurred


def get_data_for_mod_responses(responses, protocol_title):
    outputs = create_data_for_mod_responses(responses, protocol_title)
    df = final_data_for_openai(outputs)
    document_embeddings, cost_incurred = compute_doc_embeddings(df)
    return outputs, document_embeddings, cost_incurred


def create_data_for_mod_responses(responses, protocol_title):
    nheadings, ncontents, ntitles, nlinks = [], [], [], []
    outputs = []
    for response in responses:
        nheadings.append(response['question'])
        ncontents.append(response['answer'])
        ntitles.append(protocol_title + ' - mod responses')
        nlinks.append('')
    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]
    for title, h, c, t, l in zip(ntitles, nheadings, ncontents, ncontent_ntokens, nlinks):
        if (t < max_len and t > min_token_limit):
            outputs += [(title, h, c, t, l)]
        elif (t >= max_len):
            outputs += [(title, h, reduce_long(c, max_len), count_tokens(reduce_long(c, max_len)), l)]
    return outputs


# Functions to help answer queries
def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query_embedding: list[float],
                                                 contexts: dict[tuple[str, str], np.array]):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.
    Return the list of document sections, sorted by relevance in descending order.
    """
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


def construct_prompt(question: str, question_embedding: list[float], context_embeddings: dict, df: pd.DataFrame, default_answer: str):
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question_embedding,
                                                                                   context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes_string = []
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes_string.append(str(section_index))
        chosen_sections_indexes.append(section_index)

    # Useful diagnostic information
    print("Selected ", len(chosen_sections), " document sections:")
    print("\n".join(chosen_sections_indexes_string))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, 
    say """ + default_answer + """\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", chosen_sections_indexes


def answer_query_with_context(
        query: str,
        question_embedding: list,
        df: pd.DataFrame,
        document_embeddings: dict[tuple[str, str], np.array],
        default_answer: str,
        show_prompt: bool = False
):
    prompt, chosen_sections_indexes = construct_prompt(
        query,
        question_embedding,
        document_embeddings,
        df,
        default_answer
    )
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )

    # calculating the cost incurred in using OpenAI to answer the question
    answer_cost = response["usage"]["total_tokens"] * COMPLETIONS_COST / 1000

    links = []
    if len(chosen_sections_indexes) > 0 and df.loc[chosen_sections_indexes[0]]['link'] == '':
        return response["choices"][0]["text"].strip(" \n"), answer_cost, links
    for section_index in chosen_sections_indexes:
        document_section = df.loc[section_index]
        link = document_section['link']
        if link != '' and not (link in links):
            links.append(link)
        if len(links) >= 2:
            break

    return response["choices"][0]["text"].strip(" \n"), answer_cost, links