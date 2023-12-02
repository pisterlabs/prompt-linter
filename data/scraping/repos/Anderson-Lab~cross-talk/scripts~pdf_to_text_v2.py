import os
import json
import sys
import time
import openai
import numpy as np

sys.path.insert(0,'/app/')

from pyknowledgegraph import pdf

infile = sys.argv[1]
outfile = sys.argv[2]
title = sys.argv[3]

r = os.system(f'mutool convert -F text -o /app/tmp/file.txt {infile}')

if r != 0:
    print(f'mutool failed on file {infile}')
    exit(r)

openai.api_key = os.getenv("OPENAI_API_KEY")

content = open('/app/tmp/file.txt').read()
new_sentences = []
paragraphs = content.split("\n\n")
for p in paragraphs:
    new_sentences.extend(p.replace("\n"," ").replace("- ","").split("."))

import string
printable = set(string.printable)

max_tokens = 4097
batch_size = 20 # number of sentences to ask at a time
min_num_words_in_sentences = 4
min_len_sentence = min_num_words_in_sentences*4

valid_sentences = []
all_sentences = list(new_sentences)
i = 0
while i < len(all_sentences):
    gpt_prompt = f"""
Classify whether Text is a complete sentence without typos. Return the indices classified as complete as a csv.
"""
    count = 0
    tested_sentences = []
    while count < batch_size and i < len(all_sentences) and max_tokens-len(gpt_prompt) > 500:
        sent = all_sentences[i]
        i += 1
        sent = ''.join(filter(lambda x: x in printable, sent)).strip()
        num_words = len(sent.split(" "))
        if num_words < min_num_words_in_sentences or len(sent) < min_len_sentence:
            continue
        count2 = count + 1
        gpt_prompt += f"\nText {count2}. \"{sent}\""
        tested_sentences.append(sent+".")
        count += 1

    gpt_prompt += "\n\nSentences:"
    #print(gpt_prompt)
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=gpt_prompt,
      temperature=0,
      max_tokens=max_tokens-len(gpt_prompt),
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )

    for j,response in enumerate(response['choices']):
        text = response['text'].strip()
        ixs = text.split("\n")[0].replace("Text","").replace(" and ",",").replace("and","").replace(".","").split(",")
        cleaned_ixs = []
        for ix in ixs:
            try:
                if "-" in ix:
                    start,end = ix.split("-")
                    for j in range(int(start)-1,int(end)):
                        cleaned_ixs.append(j)
                else:
                    cleaned_ixs.append(int(ix)-1)
            except:
                print("Error somewhere in",text)
        valid_sentences.extend(list(np.array(tested_sentences)[cleaned_ixs]))
    #print("\n".join(valid_sentences))
    #print(gpt_prompt+text)
    #if text.startswith("Yes"):
    #    valid_sentences.append((sent+".").strip())


    print(i+1,"out of",len(all_sentences),"sentences or",int(100*(i+1)/len(all_sentences)),"percent.")
    #if i % 50 == 0:
    #    print('Sleeping')
    #    time.sleep(60)

contents = "\n".join(valid_sentences)

sections = {}
sections['title'] = title
sections['contents'] = contents
open(outfile,"w").write(json.dumps(sections))
