import time
import openai
import pandas as pd
from foi_simple import *
from tqdm import tqdm
from keys import *

openai.api_key = keys['openAI']


def gen_backstory(pid, df):
    person = df.iloc[pid]
    id = person['ids']
    backstory = BACKSTORY_START

    for k in foi_keys:
        df_val = person[k]
        elem_template = fields_of_interest[k]['template']
        elem_map = fields_of_interest[k]['valmap']

        if len(elem_map) == 0:
            newval = str(df_val)
        elif df_val in elem_map:
            newval = elem_map[df_val]

        newval = newval.replace("<1:[RECORD VERBATIM]>:", "")
        backstory += " " + elem_template.replace('XXX', newval)

    if backstory[0] == ' ':
        backstory = backstory[1:]

    return id, backstory


def do_query(prompt, max_tokens=512, engine="davinci"):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1,
        logprobs=100,
    )
    return response


# ====================================================================================

df = pd.read_csv("./NSYRdones.csv")
foi_keys = "age gender ethnicity".split()  # XXX income paredu parsedu religup
BACKSTORY_START = """When interviewed about their experiences with racism, a person who is"""
BACKSTORY_END = """ reponded with: \""""

ids = []
prompt_ids = []
prompts = []
responses = []
ethnicities = []

for pid in tqdm(range(len(df))):
    # for pid in tqdm(range(10)):
    for i in range(5):  # per count. TODO: Change back to 20

        prompt_ids.append(i)
        ethnicities.append(df.iloc[pid]['ethnicity'])
        id, prompt = gen_backstory(pid, df)
        prompt += BACKSTORY_END

        print("---------------------------------------------------")
        print(prompt)

        done = False
        while not done:
            try:
                response = do_query(prompt, max_tokens=128, engine="text-davinci-003")
                resp_text = response.choices[0]['text']
                print(resp_text)
                ids.append(id)
                prompts.append(prompt)
                responses.append(resp_text)
                done = True
            except:
                time.sleep(5.0)

newdf = pd.DataFrame(
    {'ids': ids, 'pids': prompt_ids, 'prompt': prompts, 'response': responses, 'ethnicity': ethnicities})
newdf.to_csv("./results_simple_multipass_text-davinci-003.csv")
