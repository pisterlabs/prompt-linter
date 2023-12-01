#!/usr/bin/env python

import requests
import sys
import time
import db
import openai
import hashlib
import random
import json
import re

try:
  from localsettings import *
except:
  print("Error reading localsettings")



def fix_delimeter(text, delimiter):
   list = []

   index = 0

   while(text.find(delimiter, index) != -1):
      index = text.find(delimiter, index)

# trim beginning and end of non alpha
def strip_nonalnum(word):
    return re.sub(r"^\W+|[', \"\]]+$", "", word)

def strip_heading(word):
    return re.sub(r"^([pP]ara(graph)?( )?)?\d+'?\]? ?[\:\.,] '?", "", word)

def clean_line(str):
    str2 = strip_heading(str)
    if len(str2) > 4:
        return str2
    else:
        return None
      
def get_answer(word):
    return re.sub(r"^.*\: ", "", word)

def strip_key(word):
    return re.sub(r"^.*: '?\"?", "", word)

def build_test_list(filtered_list):
    response_list = []

    i = 0
    while i < len(filtered_list) and not filtered_list[i].startswith("question"):
        i += 1

    while i + 7 <= len(filtered_list):
        item = {}
        line = filtered_list[i]
        item['question'] = strip_key(line)

        i = i + 1
        line = filtered_list[i]
        item['A)'] = strip_key(line)

        i = i + 1
        line = filtered_list[i]
        item['B)'] = strip_key(line)

        i = i + 1
        line = filtered_list[i]
        item['C)'] = strip_key(line)

        i = i + 1
        line = filtered_list[i]
        item['D)'] = strip_key(line)

        i = i + 1
        line = filtered_list[i]
        item['answer'] = strip_key(line)

        i = i + 1
        line = filtered_list[i]
        item['explanation'] = strip_key(line)

        response_list.append(item)
        i = i + 1

    return response_list

def parse_bad_json_test(filtered_list):
    response_list = []

    for line in filtered_list:
        if line.startswith("question"):
            # fix the json
            line = '{"' + line

            # try to parse it
            try:
                item = json.loads(line)
                if len(item) == 7:
                    response_list.append(line)
            except ValueError:
                nop = 1
    
    return response_list

def build_test_single_line(line):
    response_list = []

    start = line.find('{"question')
    end = line.rfind('}')
    if start != -1 and end != -1:
        line2 = line[start:end]
        line3 = '[' + line2 + '}]'
        # try to parse it
        try:
            response_list = json.loads(line3)
        except ValueError:
            nop = 1

    return response_list

def build_list_single_line(line):
    response_list = []
    str_list = line.split('", ')
    response_list = list(map(strip_nonalnum, str_list))
    response_list = list(filter(lambda x: len(x) > 4, response_list))
    return response_list

def fix_json(prompt_response):
    response_text = prompt_response['response_text']
    #print("\nparse error for prompt %s" % (prompt_response['id']))

    raw_list = response_text.split("\n")

    # remove leading and trailing non alphanumerics
    raw_list2 = list(map(strip_nonalnum, raw_list))
    
    # filter out min length of 5
    filtered_list = list(filter(lambda x: len(x) > 4, raw_list2))

    # 0 or 1 items means something is wrong
    if len(filtered_list) == 0:
        #print("\nERROR: filtered list too short for prompt %s" % (prompt_response['id']))
        response_list = []
        return response_list
    
    name = prompt_response['name']
    response_list = []

    if len(filtered_list) == 1:
        if name == "test":
            response_list = build_test_single_line(filtered_list[0])
        else:
            response_list = build_list_single_line(filtered_list[0])
    else:
        if name == "test":
            response_list = parse_bad_json_test(filtered_list)
            if len(response_list) < 2:
                response_list = build_test_list(filtered_list)
        else:
            for line in filtered_list:
                cleaned_line = clean_line(line)
                if cleaned_line != None:
                    response_list.append(cleaned_line)

    if len(response_list) > 1:
        #print("fixed it!")
        #print(response_list)
        return response_list
    else:
        print("ERROR: failed to parse prompt %s" % (prompt_response['id']))
        print(response_text)
        return None

def save_new_response(id, response_list):
    str = json.dumps(response_list)
    db.query("""UPDATE prompt_response SET response_text = %s WHERE id = %s""", [str, id])

def main(argv):
    prompt_responses = list(db.query("""SELECT r.id, r.response_text, p.name FROM prompt_response r, prompt p WHERE p.id = r.prompt_id"""))

    #print(book_ids)
    #print(prompt_ids)
    found_errors = 0
    fixed_errors = 0
    for prompt_response in prompt_responses:
        try:
            response_list = json.loads(prompt_response['response_text'])
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            found_errors += 1
            response_list = fix_json(prompt_response)

            if response_list != None:
                fixed_errors += 1
                save_new_response(prompt_response['id'], response_list)

    print("%s errors out of %s, %s fixed" % (found_errors, len(prompt_responses), found_errors))

if __name__ == "__main__":
   main(sys.argv[1:])

"""
str = list(db.query("SELECT * FROM prompt_response WHERE id = 16595917599065866207"))[0]['response_text']
str_list = str.split("\n")

str_list2 = list(map(strip_nonalnum, str_list))

# filter out min length of 5
filtered_list = list(filter(lambda x: len(x) > 4, str_list2))


# split commas not inside quotes
newStr = re.split(r',(?=")', str)

# trim beginning and end of non alpha
def strip_nonalnum(word):
    return re.sub(r"^\W+|[', \"\]]+$", "", word)

def strip_heading(word):
    return re.sub(r"^([pP]ara(graph)?( )?)?\d+'?\]? ?[\:\.] '?", "", word)


# remove stuff outside outer quotes?
re.search(r"\"(.*)\"", str)

For '    "Paragraph 3: Ridle'... 
m = re.search(r"^.{0,10}aragraph \d+\: (.*)\"", str2)
m.group(1)
if m != None and len(m.groups()) == 1:
    clean = m.group()[0]

# different logic for tests
# split by newlines
# filter out short lines
# ensure divisible by 7
# get answers and assume question / a / b / c / d / answer / explanation order
def get_answer(word):
    return re.sub(r"^.*\: ", "", word)

"""