from asgiref.sync import sync_to_async
import discord
from transformers import GPT2TokenizerFast
import sys

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
import urllib
import requests
import json
import multiprocessing
import ssl
import gspread
from itertools import islice

from gitbook_scraper import * # import from the self_run_bot folder since we have removed gitbook_scraper.py from this folder
from pdf_parse_seq import * # import from the self_run_bot folder since we have removed pdf_parse_seq.py from this folder

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
config = dotenv_values("../.env")

openai.api_key = config['OPENAI_API_KEY']

TELEGRAM_TOKEN = config['TELEGRAM_TOKEN']
TELEGRAM_API_URL = "https://api.telegram.org/bot{}/".format(TELEGRAM_TOKEN)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "cl100k_base"
min_token_limit = 10

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

# f"Context separator contains {separator_len} tokens"


protocol_id = "1"
bot_description = "Router protocol relayer info"


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

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

def read_docs(github_repo):
    g = Github(config['GITHUB_ACCESS_TOKEN'])
    repo = g.get_repo(github_repo)
    title_stack=[]
    contents = repo.get_contents("")
    while contents:
        try:
            file_content = contents.pop(0)
        except Exception:
            pass
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            if file_content.path.find('relayer') == -1: ## remove this line later
                continue ## remove this line later
            if file_content.name.endswith('md') or file_content.name.endswith('mdx'):
                file_contents = repo.get_contents(file_content.path)
                title = pp.AtLineStart(pp.Word("#")) + pp.rest_of_line
                sample = file_contents.decoded_content.decode()
                sample = cleanup_data(sample)

                title_stack.append([0, 'start_of_file'])
                if sample.split('\n')[0] == '---':
                    title_stack[-1].append('')
                    title_stack[-1].append(file_content.path)
                    title_stack.append([1, sample.split('\n')[1].split(':')[1].lstrip()])
                    sample = sample.split('---')[2]

                last_end = 0
                for t, start, end in title.scan_string(sample):
                    # save content since last title in the last item in title_stack
                    title_stack[-1].append(sample[last_end:start].lstrip("\n"))
                    title_stack[-1].append(file_content.path)

                    # add a new entry to title_stack
                    marker, title_content = t
                    level = len(marker)
                    title_stack.append([level, title_content.lstrip()])

                    # update last_end to the end of the current match
                    last_end = end

                # add trailing text to the final parsed title
                title_stack[-1].append(sample[last_end:])
                title_stack[-1].append(file_content.path)
    return title_stack

def create_data_for_docs(protocol_title, title_stack):
    heads = {}
    max_level = 0
    nheadings, ncontents, ntitles = [], [], []
    outputs = []
    max_len = 1500
    s1 = '<Section'
    s2 = '</Section>'

    for level, header, content, dir in title_stack:
        final_header = header
        dir_elements = dir.split('/')
        element_len = 1
        dir_header = ''
        sub = 1
        title = protocol_title + " - " + dir_elements[0]
        if dir_elements[len(dir_elements) - sub].find('README') != -1:
            sub = sub + 1
        while element_len < len(dir_elements) - sub:
            dir_header = dir_header + dir_elements[element_len] + ': '
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
            i=i-1
        final_header = dir_header + final_header
        if final_header.find('start_of_file') == -1:
            remove_content = find_between(content, s1, s2)
            content = content.replace(remove_content,'').replace(s1, '').replace(s2, '')
            if content.strip() == '':
                continue
            nheadings.append(final_header.strip())
            ncontents.append(content)
            ntitles.append(title)

    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]
    # outputs += [(title, h, c, t) if t<max_len
    #             else (title, h, reduce_long(c, max_len), count_tokens(reduce_long(c,max_len)))
    #             for title, h, c, t in zip(ntitles, nheadings, ncontents, ncontent_ntokens)]
    for title, h, c, t in zip(ntitles, nheadings, ncontents, ncontent_ntokens):
        if (t<max_len and t>min_token_limit):
            outputs += [(title,h,c,t)]
        elif(t>=max_len):
            outputs += [(title, h, reduce_long(c, max_len), count_tokens(reduce_long(c,max_len)))]
    return outputs

def final_data_for_openai(outputs):
    res = []
    res += outputs
    df = pd.DataFrame(res, columns=["title", "heading", "content", "tokens"])
    df = df[df.tokens > 10]   # to ensure really small and insignificant data doesn't get indexed
    df = df.drop_duplicates(['title','heading'])
    df = df.reset_index().drop('index',axis=1) # reset index
    df = df.set_index(["title", "heading"])
    return df


def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.rindex( last, start )
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

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    time.sleep(5)
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
        (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[tuple[str, str], np.array]) -> list[tuple[float, tuple[str, str]]]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print("Selected ", len(chosen_sections), " document sections:")
    print("\n".join(chosen_sections_indexes))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def answer_query_with_context(
		query: str,
		df: pd.DataFrame,
		document_embeddings: dict[tuple[str, str], np.array],
		show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
		query,
		document_embeddings,
		df
	)
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
		prompt=prompt,
		**COMPLETIONS_API_PARAMS
	)

    return response["choices"][0]["text"].strip(" \n")

def read_from_github(protocol_title, github_repo):
    title_stack = read_docs(github_repo)
    outputs = create_data_for_docs(protocol_title, title_stack)
    print(outputs)
    df = final_data_for_openai(outputs)
    print(df.head)
    df = df.set_index(["title", "heading"])
    document_embeddings = compute_doc_embeddings(df)
    print(len(df), " rows in the data.")
    return outputs, df, document_embeddings

def calculate_new_output(title, heading, content):
    nheadings, ncontents, ntitles = [], [], []
    outputs = []
    max_len = 1500
    nheadings.append(heading)
    ncontents.append(content)
    ntitles.append(title)
    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]

    for title, h, c, t in zip(ntitles, nheadings, ncontents, ncontent_ntokens):
        if (t<max_len and t>min_token_limit):
            outputs += [(title,h,c,t)]
        elif(t>=max_len):
            outputs += [(title, h, reduce_long(c, max_len), count_tokens(reduce_long(c,max_len)))]

    return outputs

def add_data(protocol_id, title, heading, content):
    outputs, embeddings = retrieve_from_db(protocol_id)
    # check to ensure that the output does not already include this entry
    for x in outputs:
        if(title == x[0] and heading == x[1] and content == x[2]):
            print("Data already present")
            return "Data already present"
    # take title, heading and content and fetch the new outputs
    new_outputs = calculate_new_output(title, heading, content)
    new_df = final_data_for_openai(new_outputs)
    # create an embedding against the newly added data
    new_document_embeddings = compute_doc_embeddings(new_df)
    # append the new output to the outputs in the database
    outputs.extend(new_outputs)
    # append the new embedding to the embedding in the database
    embeddings.update(new_document_embeddings)
    return update_in_db(outputs, embeddings, protocol_id)

def update_data(protocol_id, title, heading, updated_content):
    outputs, embeddings = retrieve_from_db(protocol_id)
    index_to_delete = -1
    for i, x in enumerate(outputs):
        if(title == x[0] and heading == x[1]):
            if updated_content == x[2]:
                return "Updated data already present"
            else:
                index_to_delete = i
    if(index_to_delete > 0):
        new_outputs = calculate_new_output(title, heading, updated_content)
        new_df = final_data_for_openai(new_outputs)
        # create an embedding against the newly added data
        new_document_embeddings = compute_doc_embeddings(new_df)
        # append the new output to the outputs in the database
        outputs.extend(new_outputs)
        # append the new embedding to the embedding in the database
        embeddings.update(new_document_embeddings)
        # deleting the existing entry
        updated_outputs, updated_embeddings = delete_entries_by_index(outputs, embeddings, index_to_delete)
    return update_in_db(updated_outputs, updated_embeddings, protocol_id)

def delete_data(protocol_id, title, heading):
    outputs, embeddings = retrieve_from_db(protocol_id)
    index_to_delete = -1
    for i, x in enumerate(outputs):
        if(title == x[0] and heading == x[1]):
            index_to_delete = i
    if index_to_delete<0:
        return "Title and heading not found"
    updated_outputs, updated_embeddings = delete_entries_by_index(outputs, embeddings, index_to_delete)
    return update_in_db(updated_outputs, updated_embeddings, protocol_id)

def delete_entries_by_index(outputs, embeddings, index):
    outputs.pop(index)
    # del embeddings[next(islice(embeddings, index, None))]
    return outputs, embeddings

def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content

def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js

def get_updates(offset=None):
    url = TELEGRAM_API_URL + "getUpdates?timeout=100"
    if offset:
        url += "&offset={}".format(offset)
    js = get_json_from_url(url)
    return js

def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)

def echo_all(updates, df, document_embeddings):
    for update in updates["result"]:
        try:
            text = update["message"]["text"]
            chat = update["message"]["chat"]["id"]
            send_message(text, chat, df, document_embeddings)
        except Exception as e:
            print(e)

def send_message(text, chat_id, df, document_embeddings):
    print(text)
    response = answer_query_with_context(text, df, document_embeddings)
    url = TELEGRAM_API_URL + "sendMessage?text={}&chat_id={}".format(response, chat_id)
    get_url(url)

class TelegramBot(multiprocessing.Process):
    def __init__(self, df, document_embeddings):
        super(TelegramBot, self).__init__()
        self.df = df
        self.document_embeddings = document_embeddings

    def run(self):
        print("I'm the process with id: {}".format(self.df))
        last_update_id = None
        while True:
            updates = get_updates(last_update_id)
            if len(updates["result"]) > 0:
                last_update_id = get_last_update_id(updates) + 1
                echo_all(updates, self.df, self.document_embeddings)
            time.sleep(0.5)

class DiscordBot(multiprocessing.Process):
    def __init__(self, df, document_embeddings):
        super(DiscordBot, self).__init__()
        self.df = df
        self.document_embeddings = document_embeddings

    def run(self):
        print("I'm the process with id: {}".format(self.df))
        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            print('We have logged in as {0.user}'.format(client))

        @client.event
        async def on_message(message):
            print(message.content)
            if message.author == client.user:
                return
            
            if message.content.lower().find(config['DISCORD_BOT_USER_ID'].lower()) != -1:
            # replace @scarlett with the Discord name of your bot
                answer = answer_query_with_context(message.content, self.df, self.document_embeddings)
                await message.channel.send(answer)
                question = message.content.replace('<' + config['DISCORD_BOT_USER_ID'] + '> ', '')
                send_question_to_db(protocol_id, question, answer)

        client.run(config['DISCORD_TOKEN'])

def untuplify_dict_keys(mapping):
    string_keys = {json.dumps(k): v for k, v in mapping.items()}
    return string_keys

def tuplify_dict_keys(string):
    mapping = string
    return {tuple(json.loads(k)): v for k, v in mapping.items()}

def send_question_to_db(protocol_id, question, answer):
    data_to_post = {"protocol_id": protocol_id, "question": question, "answer": answer}
    response = requests.post(config['BASE_API_URL'] + "question/", json=data_to_post)
    return response

# adding data to the database
def send_to_db(_id, description, outputs, document_embeddings):
    data_to_post = {"_id": _id, "description": description, "data": outputs, "embeddings": untuplify_dict_keys(document_embeddings)}
    response = requests.post(config['BASE_API_URL'] + "document/", json=data_to_post)
    return response


# fetching outputs and embeddings from the database
def retrieve_from_db(_id):
    response = requests.get(config['BASE_API_URL'] + "document/" + _id)
    json_response = response.json()
    outputs = json_response['data']
    document_embeddings = tuplify_dict_keys(json_response['embeddings'])
    return outputs, document_embeddings

def update_in_db(outputs, embeddings, _id):
    # update the entry in the database
    data_to_update = {"data": outputs, "embeddings": untuplify_dict_keys(embeddings)}
    response = requests.put(config['BASE_API_URL'] + "document/" + _id, json=data_to_update)
    return response
        
# to add data from google sheet        
def add_data_from_sheet(protocol_id, sheet_id, sheet_name):
    gc = gspread.service_account('PATH_TO_CREDENTIALS.JSON')
    spreadsheet = gc.open_by_key(sheet_id)
    worksheet = spreadsheet.worksheet(sheet_name)
    rows = worksheet.get_all_records()
    print(rows)
    df = pd.DataFrame(rows)
    data_dict = {}
    upload_df = pd.DataFrame()
    for index, data in df.iterrows():
        if data['Uploaded'] == 'No' or data['Uploaded'] == '':
            # Upload to df and embeddings
            add_data(protocol_id, data['Title'], data['Heading'], data['Content'])

            # Recreate the df to upload back to the gsheet
            data_dict['Title'] = data['Title']
            data_dict['Heading'] = data['Heading']
            data_dict['Content'] = data['Content']
            data_dict['Uploaded'] = 'Yes'
            print(data_dict)
            data_df = pd.DataFrame([data_dict])
            upload_df = pd.concat([upload_df, data_df])            

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

def get_data_from_gitbook(gitbook_data_type, gitbook_link):
    content = get_gitbook_data_in_md_format(gitbook_link, '')
    print('Gitbook data in md format fetched')
    title_stack = add_data_array(gitbook_data_type, content)
    outputs = create_data_for_docs(title_stack)
    print('Outputs created for gitbook data')
    df = final_data_for_openai(outputs)
    print(df.head)
    df = df.set_index(["title", "heading"])
    document_embeddings = compute_doc_embeddings(df)
    print('Embeddings created, sending data to db...')
    return outputs, df, document_embeddings

def get_pdf_data(type, document):
    content = convert_to_md_format(document)
    title_stack = add_data_array(type, content)
    outputs = create_data_for_docs(title_stack)
    print('Outputs created for PDF data')
    df = final_data_for_openai(outputs)
    print(df.head)
    df = df.set_index(["title", "heading"])
    document_embeddings = compute_doc_embeddings(df)
    print('Embeddings created, sending data to db...')
    return outputs, df, document_embeddings

def read_command_line_params():
    arguments = {}
    n = len(sys.argv)

    print("\nArguments passed:", end = " ")
    for i in range(1, n):
        argument = sys.argv[i].split('=')
        arguments[argument[0]] = argument[1]
    return arguments

def main():
    """Parameters to provide from the command line
    protocol_id (string): protocol_id for which the data needs to be uploaded
    protocol_description (string) : description of the protocol for which data is being uploaded
    reset_bot | first_start : if you're uploading the data for the first time or if you want upload entirely new data and forget old data
    read_from_github (boolean) : true if data needs to be read from Github, else false
    github_repo (string) : link to your Github repo containing the docs, mandatory if read_from_github is true (for eg. https://github.com/router-protocol/router-chain-docs)
    protocol_title : protocol name, used while uploading data from Github
    read_from_gitbook_link (boolean): true if data needs to be read from Gitbook docs, else false
    gitbook_link (string) : link to your Gitbook docs, mandatory if read_from_gitbook_link is true
    gitbook_data_type : the type of data being uploaded from Gitbook - whitepaper, dev docs, FAQs etc.
    read_from_pdf : true if data needs to be read from a PDF document, otherwise false
    pdf_path : link to the PDF, mandatory if read_from_pdf is true
    read_from_sheet: true if data needs to be read from a Google sheet, else false
    sheet_id: ID of the Google Sheet from which the data is to be read, mandatory if read_from_sheet is true
    sheet_name: name of the spreadsheet from which the data is to be read, mandatory if read_from_sheet is true
    """

    arguments = read_command_line_params()
    protocol_id = arguments['protocol_id']
    bot_description = arguments['bot_description']
    protocol_title = arguments['protocol_title']
    outputs = []

    # if the user is uploading data for the first time or wants to reupload the data from scratch
    if arguments['reset_bot'].lower == 'true' or arguments['first_start'].lower == 'true':
        if arguments['read_from_github'].lower == 'true':
            try:
                github_repo = arguments['github_repo']
            except KeyError:
                raise Exception("Github link not provided, while read from github is true")
            outputs, df, document_embeddings = read_from_github(protocol_title, github_repo)
            if len(outputs) > 0:
                send_to_db(protocol_id, bot_description, outputs, document_embeddings)

        if arguments['read_from_gitbook_link'].lower == 'true':
            try:
                gitbook_link = arguments['gitbook_link']
            except KeyError:
                raise Exception("Gitbook link not provided, while read_from_gitbook_link is true")
            try:
                gitbook_data_type = arguments['gitbook_data_type']
            except KeyError:
                gitbook_data_type = 'Whitepaper'
            outputs, df, document_embeddings = get_data_from_gitbook(gitbook_data_type, gitbook_link)
            if len(outputs) > 0:
                send_to_db(protocol_id, bot_description, outputs, document_embeddings)

        if arguments['read_from_pdf'].lower == 'true':
            try:
                pdf_path = arguments['pdf_path']
            except KeyError:
                raise Exception("PDF path not provided, while read_from_pdf is true")
            outputs, df, document_embeddings = get_pdf_data('PDF', pdf_path)
            if len(outputs) > 0:
                send_to_db(protocol_id, bot_description, outputs, document_embeddings)


    if arguments['read_from_sheet'].lower == 'True'.lower():
        sheet_id = arguments['sheet_id']
        sheet_name = arguments['sheet_name']
        add_data_from_sheet(protocol_id, sheet_id, sheet_name)


    # p = TelegramBot(df, document_embeddings)
    # p.start()
    outputs_from_database, document_embeddings_from_database = retrieve_from_db(protocol_id)
    df_from_database = final_data_for_openai(outputs_from_database)
    p = DiscordBot(df_from_database, document_embeddings_from_database)
    p.start()
    
    


if __name__ == '__main__':
    main()