# collect data for gpt-3 aut pilot
# 30 respondents, instruction 2, presence and frequency penalties 1
# temperature range .65 - .75
# length of at least 33 words -> choose max_tokens 64
# median human fluency in experiment was 9 responses so use this

# import libraries
import numpy as np
import csv
from datetime import datetime
import os
import openai

# filename to write data to
csv_file = '220422_gpt3_aut_pilot.csv' 

# load API key from environment variable 
openai.api_key = os.getenv("OPENAI_API_KEY")

# fixed parameters to supply each run
max_tokens = 80
n = 1 
engine = "text-davinci-002"

# GPT-3 variables and parameters to optimize
temperature = np.arange(.67, .77, .01)
presence_penalty = 1
frequency_penalty = 1

# AUT variables Instructions to try
aut_object = ['book', 'fork', 'tin can']
instr = 'What are some creative uses for a {}? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. List 9 creative uses for a {}.'

# function to call gpt3 api with instr and parameter settings
def gpt3_response(eng, instr, max_tok, temp, pres_pnty, freq_pnty, num):
    response = openai.Completion.create(
        engine=eng, 
        prompt=instr, 
        max_tokens=max_tok,
        temperature=temp,
        presence_penalty=pres_pnty,
        frequency_penalty=freq_pnty,
        n=num)
    return(response)

# open csv file to write data to
f = open(csv_file, 'w')
# create csv writer
writer = csv.writer(f)
# write header to csv file
header = ['id', 'gpt3_id', 'gpt3_created', 'gpt3_model', 'engine', 'timestamp', 'temperature', 'presence_penalty', 'frequency_penalty', 'instr_text', 'aut_object', 'gpt3_response']
writer.writerow(header)

# loop through all variables / parameters of the experiment
rowid = 0
for aut_obj in aut_object:
    for temp in temperature:
        for i in range(3):
            instr_text = instr.format(aut_obj, aut_obj) 
            response = gpt3_response(engine, instr_text, max_tokens, temp, presence_penalty, frequency_penalty, n)
            rowid = rowid + 1
            gpt3_response_text = response['choices'][0]['text']
            gpt3_id = response['id']
            gpt3_created = response['created']
            gpt3_model = response['model']
            row = [rowid, gpt3_id, gpt3_created, gpt3_model, engine, datetime.now(), temp, presence_penalty, frequency_penalty, instr_text, aut_obj, gpt3_response_text]
            writer.writerow(row)


# close csv writer and file
f.close()
