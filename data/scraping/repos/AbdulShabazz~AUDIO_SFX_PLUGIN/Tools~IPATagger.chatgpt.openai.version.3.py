import os
import re
import json
import time as time
import openai
import threading

archive = {}
completed_archive = {}
todo_archive = {}

def populate_todo_archive():
    RESUME_KEYWORD = ""
    resume_ready = True
    with open("database.corpus.generated.optimized.log","r") as f:
        for wd in f.readlines():
            if not resume_ready and re.search("^"+RESUME_KEYWORD, wd):
                resume_ready = True
            if resume_ready and not re.search("^#", wd):
                wd = re.sub("[\'\.]+", "", wd)
                todo_archive[wd] = True

def populate_archive(): # the latest archive
    with open("pronounciation.corpus.generated.1674299232.2774985.log","r") as g:
        for wd in g.readlines():
            wd = wd.lower()
            completed_archive[wd] = True
            word = re.sub("[_]+", "", wd)
            if word != wd and word in completed_archive: # remove unformatted dups
                completed_archive.pop(word)
            archive[word] = True

th_0000 = threading.Thread(target=populate_todo_archive)
th_0001 = threading.Thread(target=populate_archive)

th_0000.start()
th_0001.start()

th_0000.join()
th_0001.join()

todos = []

with open("pronounciation.corpus.todo.generated.log","w") as f:
    for word in dict.keys(todo_archive):
        if word not in archive:
            todos += [word]
            f.writelines(word) # preserve newlines

myprompt = '''(1) Seperate each word by a newline, and (2) split each word into syllables -- separated by underscores: '''

# OPENAI
# @param - prompt [prompt.length + response.length < max_tokens]
# @param - engine [code/text]-[davinci/curie/babbage/ada]-[ver]
# @param - max_tokens [prompt + response < 4097] Length. Bandwidth alloted for the session/connection 
# @param - temperature [1.0 - 0.0] Strictness. Controls the creativity or randomness of the response
# @param - top_p [1.0 - 0.0] Predictability. Controls the constrained randomness of oncomming text
# @param - frequency_penalty [1.0 - 0.0] Readability. Controls the ratio of used uncommon words
# @param - presence_penalty [1.0 - 0.0] DisAllowed repetitiveness/Use synonyms. Ratio of input tokens allowed in the response
# @returns  - { choices[{ engine:[davinci, curie], finish_reason:[stop,length], index:N, logprob:[], text:[response]},] }
def issueQuery(word,g):
    openai.api_key = os.getenv("OPENAI_API_KEY") # Set before callig this module
    response = openai.Completion.create(model="text-davinci-003", max_tokens=1600, presence_penalty=0.0, top_p=1.0, temperature = 1.0, prompt=myprompt + json.dumps(word) + "?")
    for choice in response.choices:
        #result = word + " = " + choice.text
        g.writelines(choice.text.lower() + "\n")
        #print(choice.text)

MAX_TOKENS = 4097 # total syllables = prompt + completions
MAX_SEND_TOKENS = 425
MAX_THREADS = 18
SINGLE_THREADED = False

with open("pronounciation.corpus.generated." + str(time.time_ns()) + ".log","w") as g:
    alltasks = []
    worker = []
    for word in todos:
        try:
            if len(worker) < MAX_SEND_TOKENS:
                worker += [re.sub("[\n]+","",word)]
            else:
                if SINGLE_THREADED:
                    issueQuery(worker,g)
                else:
                    time.sleep(0.001)
                    th = threading.Thread(target=issueQuery, args=(worker,g))
                    th.start()
                    alltasks += [th]
                worker = []
        except Exception as e:
            print(e)
            #time.sleep(1)
    if len(worker):
        issueQuery(worker,g)
        worker = []
    if len(alltasks):
        for th in alltasks:
            th.join()
    for word in dict.keys(completed_archive):
        g.writelines(word.lower()) # preserve newlines