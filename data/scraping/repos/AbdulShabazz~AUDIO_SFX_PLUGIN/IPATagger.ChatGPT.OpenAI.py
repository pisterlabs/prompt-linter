import os
import re
import json
import time as time
import openai
import threading

def issueQuery(word,g):
    time.sleep(1.1)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(model="text-davinci-003", prompt="What are the IPA spelling for " + json.dumps(word) + "?")
    #buff = []
    #for choice in response.choices: 
    #    buff += ["_".join(list(re.sub("[\n/]+", "", choice.text)))]
    #statement = "  " + word + " : " + json.dumps(buff) + ",\n"
    #statement = "  " + word + " : " + "_".join(list(re.sub("[\n/]+", "", response.choices[0].text)))
    #g.writelines(statement+"\n")
    #print(statement)
    print(response.choices)

MAXTHREADS = 18
RESUME_KEYWORD = "125th"
resume = True
worker = []
with open("database.corpus.generated.optimized.log","r") as f:
    with open("pronounciation.corpus.generated.version.1.log","a") as g:
        for wd in f.readlines():
            if not resume and re.search("^"+RESUME_KEYWORD, wd):
                resume = True
            if resume:
                word = re.sub("[\'\.\n]+", "", wd)
                try:
                    #th = threading.Thread(target=issueQuery, args=(wd,g,))
                    #th.start()
                    #worker += [th]
                    if len(worker) < MAXTHREADS:
                        worker += [word]
                    else:
                        issueQuery(worker,g)
                        worker = []
                except Exception as e:
                    print(e)

#for th in worker:
#    th.join()

#with open("pronounciation.corpus.generated.log","w") as g:
#    g.writelines(json.dumps(syllable, sort_keys=True, indent=2))