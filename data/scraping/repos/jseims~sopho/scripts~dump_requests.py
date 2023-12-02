#!/usr/bin/env python

import sys
import time
import db
import openai
import hashlib
import random
import json

OPENAI_API_KEY = ''
OPENAI_CHAT_MODEL = ''
DEFAULT_PROMPT_SET = 0
try:
  from localsettings import *
except:
  print("Error reading localsettings")

# 1 for top level summary, 2 for next summary, 3 for all the rest
REQUEST_LEVEL = 3


def dictValuePad(key):
    return '%(' + str(key) + ')s'

def insertFromDict(table, dict):
    """Take dictionary object dict and produce sql for 
    inserting it into the named table"""
    sql = 'INSERT INTO ' + table
    sql += ' ('
    sql += ', '.join(dict)
    sql += ') VALUES ('
    sql += ', '.join(map(dictValuePad, dict))
    sql += ');'
    return sql


def filter_prompts(prompts, level):
    level_prompts = []

    for prompt in prompts:
        prompt_level = int(prompt.get("level"))
        if prompt_level == level:
            level_prompts.append(prompt)

    return level_prompts


def response_exists(hash):
    # lookup by hash
    sql = "SELECT * FROM prompt_response WHERE prompt_hash = %s"
    args = [hash]
    rows = db.query(sql, args, False)
    return len(rows) != 0



def make_request(book_id, title, author, parent_item, name, prompt):
    request = None
    prompt_text = prompt.get("prompt_text")
    #print(prompt_text)
    if parent_item == None:
        prompt_text = prompt_text % (title, author)
        parent_id = -1
    else:
        parent_id = parent_item['id']
        if name == "test":
            answer_letter = parent_item['text']['answer']
            answer = parent_item[answer_letter]
            prompt_text = prompt_text % (title, author, parent_item["question"], answer)
            #print(prompt_text)
        else:
            prompt_text = prompt_text % (title, author, parent_item['text'])

    system = prompt.get("system_text")

    bstring = bytes("%s%s%s" % (prompt.get("id"), system, prompt_text), 'utf-8')
    hash_object = hashlib.sha256(bstring)
    hex_dig = hash_object.hexdigest()
    if not response_exists(hex_dig):
        request = {
        "messages": [{"role": "system", "content": system},
                        {"role": "user", "content": prompt_text}],
        "metadata" : {'start_time': time.time(), 'book_id' : book_id, 'prompt_id' : prompt.get("id"), 'hash' : hex_dig, 'system' : system, 'prompt_text' : prompt_text, 'model' : OPENAI_CHAT_MODEL, 'response_piece_id' : parent_id}}
    else:
        print("response already exists, skipping prompt id %s\n" % (prompt.get("id")))
        nop = 1
    return request



def get_response_list(prompt_response_id, name):
    response_pieces = list(db.query("""SELECT * from response_piece WHERE prompt_response_id = %s""", [prompt_response_id]))
    #response_list =[]

    #for piece in response_pieces:
    #    response_list.append(piece['text'])

    #json_obj = json.loads(text)

    return response_pieces

def make_top_requests_list():
    books = list(db.query("""SELECT id, title, author from book"""))
    prompts = list(db.query("""SELECT * from prompt WHERE prompt_set = %s AND level = 1""", [DEFAULT_PROMPT_SET]))
    request_list = []
    
    for book in books:
        for prompt in prompts:
            request = make_request(book['id'], book['title'], book['author'], None, prompt['name'], prompt)
            if request != None:
                request_list.append(request)
    
    return request_list


def make_deep_requests_list(level):
    sql = """select r.id, r.book_id, r.prompt_id, p.name, b.title, b.author from prompt_response r, prompt p, book b where r.prompt_id = p.id and r.book_id = b.id and p.level = %s AND p.prompt_set = %s AND b.id != 3"""
    args = [level - 1, DEFAULT_PROMPT_SET]
    prompts = list(db.query(sql, args))
    request_list = []

    for prompt in prompts:
        #print(prompt)
        prompt_response_id = prompt['id']
        book_id = prompt['book_id']
        title = prompt['title']
        author = prompt['author']
        prompt_id = prompt['prompt_id']
        name = prompt['name']
        
        response_list = get_response_list(prompt_response_id, name)
        #print(response_list)

        child_prompts = list(db.query("""SELECT * FROM prompt WHERE parent_id = %s""", [prompt_id]))

        for item in response_list:
            for child_prompt in child_prompts:
                request = make_request(book_id, title, author, item, name, child_prompt)
                if request != None:
                    request_list.append(request)

    return request_list

def dump_request(request, file):
    data = {"model" : OPENAI_CHAT_MODEL, "messages" : request["messages"], "metadata" : request["metadata"]}

    json_string = json.dumps(data)
    file.write(json_string + "\n")


def main(argv):
    request_list = []
    if REQUEST_LEVEL == 1:
        request_list = make_top_requests_list()
    else:
        request_list = make_deep_requests_list(REQUEST_LEVEL)

    print("starting %s requests" % len(request_list))
    with open("requests.json", "w") as f:
        for request in request_list:
            print("doing it")
            dump_request(request, f)
            print("done")
        f.close()
    print("done")


if __name__ == "__main__":
   main(sys.argv[1:])

#{"model": "text-embedding-ada-002", "input": "0\n", "metadata": {"row_id": 1}}