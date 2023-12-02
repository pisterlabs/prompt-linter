import os
import re
import json
import time as time
import openai
import threading
import IPAPronunciation.corpus.todo.generator.class as corpus_todos

corpus = corpus_todos.update_todos()

corpus.update()

archive = corpus.archive
completed_archive = corpus.completed_archive
todo_archive = corpus.todo_archive
todos = corpus.todos

myprompt = '''(1) Seperate each word by a newline, and (2) split each word into syllables -- separated by underscores: '''

# OPENAI
# @param - prompt [prompt.length + response.length < max_tokens]
# @param - engine [gpt-3.5-turbo/gpt-4(13pg max/gpt-4-32k(52pg max))] or [[code/text]-[davinci/curie/babbage/ada]-[ver]]
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

with open("pronunciation.corpus.generated." + str(time.time_ns()) + ".log","w") as g:
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