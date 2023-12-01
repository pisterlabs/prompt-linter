import re
import openai
import json
import time
import numpy as np

openai.api_key = ""

def embed(s):
    max_retries = 10
    retry_delay = 5  # seconds

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = openai.Embedding.create(
                        input=s,
                        model="text-embedding-ada-002"
                    )
            return np.asarray(response['data'][0]['embedding'])
        except Exception as e:
            print(f"Error occurred: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                break  # Successful execution, exit the loop

    if retry_count == max_retries:
        print("Max retries reached. Exiting.")
        raise TimeoutError

def embed_trp(triple):
  assert len(triple) == 3
  return np.concatenate((embed(triple[0]), embed(triple[1]), embed(triple[2]), embed(' '.join(triple))))

def cos_dist(x,y):
  return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

def dist(triple_a, triple_b, weight=0.5):
  """weight should be in (0,1)"""
  
  a = cos_dist(triple_a[0:4608], triple_b[0:4608])
  b = cos_dist(triple_a[4608:], triple_b[4608:])
  
  return a*(1-weight) + b*weight

def extract(text):
    matches = re.findall(r'\((.*?)\)', text)

    out = []
    for match in matches:
        out.append(match.split(", "))

    return out

def extract_json(text):

    try:
        new_text = re.sub(r'\d\.', '', text)
        new_text = re.sub(r'\n\n', ', ', new_text)
        
        js = json.loads('[' + new_text + ']')
        out = []

        for triple in js:
            out.append([triple["subject"], triple["predicate"], triple["object"]])
        return out

    except:
        print("Could not parse {}".format(text))
        return []

def get_chat_gpt_triples(sentence, MODEL):

    content = "Extract all semantric triples from sentence, output JSON only with a list of triples, with keys 'subject', 'predicate', 'object' for each triple: {}".format(sentence)
    
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": content},
    ],
    temperature=0,
    )

    print(response)

    return extract(response['choices'][0]['message']['content']), response

def two_shot_chatgpt(sentence):
    max_retries = 10
    retry_delay = 5  # seconds

    retry_count = 0
    while retry_count < max_retries:
        try:
            return few_shot_get_chat_gpt_triples(sentence)
        except Exception as e:
            print(f"Error occurred: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                break  # Successful execution, exit the loop

    if retry_count == max_retries:
        print("Max retries reached. Exiting.")
        raise TimeoutError

def one_shot_chatgpt(sentence, mem):
    if sentence in mem:
        print("Cache used.")
        return mem[sentence]

    max_retries = 10
    retry_delay = 5  # seconds

    retry_count = 0
    while retry_count < max_retries:
        try:
            res =  one_shot_get_chat_gpt_triples(sentence)
            mem[sentence] = res
            return res

        except Exception as e:
            print(f"Error occurred: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                break  # Successful execution, exit the loop

    if retry_count == max_retries:
        print("Max retries reached. Exiting.")
        raise TimeoutError

def few_shot_get_chat_gpt_triples(sentence, MODEL="gpt-3.5-turbo"):

    def trp(sub, pred, obj):
        return "{\"subject\": \"" + sub + "\", \"predicate\": \"" + pred + "\", \"object\": \"" + obj + "\"}"

    intro = "Extract all semantric triples from sentence."

    sent_A = "Sentence: These tracks have subsequently been included on CD reissues of the album `` The Plan ''."
    triples_A = "Triples: [{}, {}]".format(trp("tracks", "have been included", "on CD reissues of ``The Plan''"),
                                           trp("tracks", "have been included on CD reissues of", "album"))

    sent_B = "Sentence: He and his friends were said to have made bombs for fun on the outskirts of Murray , Utah ."
    triples_B = "Triples: [{}, {}, {}, {}, {}, {}]".format(trp('He', 'were said to', 'have made bombs'), 
                                                trp('He', 'were said to have made bombs on outskirts of Murray', 'for fun'),
                                                trp('his friends', 'were said to', 'have made bombs'), 
                                                trp('his friends', 'were said to have made bombs on outskirts of Murray', 'for fun'),
                                                trp('He', 'were said to made bombs on outskirts of', 'Murray'),
                                                trp('his friends', 'were said to made bombs on outskirts of', 'Murray') )

    content = intro + " \n " + sent_A + " " + triples_A + " \n " + sent_B + " " + triples_B + " \n " + "Sentence: {}".format(sentence) + " Triples:"

    print(content)

    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": content},
    ],
    temperature=0,
    )

    print(response)

    return response


def one_shot_get_chat_gpt_triples(sentence, MODEL="gpt-3.5-turbo"):

    def trp(sub, pred, obj):
        return "{\"subject\": \"" + sub + "\", \"predicate\": \"" + pred + "\", \"object\": \"" + obj + "\"}"

    intro = "Extract all semantric triples from sentence."

    sent_A = "Sentence: These tracks have subsequently been included on CD reissues of the album `` The Plan ''."
    triples_A = "Triples: [{}, {}]".format(trp("tracks", "have been included", "on CD reissues of ``The Plan''"),
                                           trp("tracks", "have been included on CD reissues of", "album"))

    content = intro + " \n " + sent_A + " " + triples_A + " \n " + "Sentence: {}".format(sentence) + " Triples:"

    print(content)
    
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": content},
    ],
    temperature=0,
    )

    print(response)

    return response

def get_benchie_sentences():

    benchie_sentences = []

    with open('./sample300_en.txt', 'r') as file:
        for line in file:
            benchie_sentences.append(line.strip())

    return benchie_sentences

def reformat_output(text):
    MODEL = "gpt-3.5-turbo"    
    content = 'Rewrite semantic triples to JSON format: ' + text
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": content},
    ],
    temperature=0,
    )

    print(response)

    return extract_json(response['choices'][0]['message']['content']), response

"""
benchie_sentences = get_benchie_sentences() 

out = {}
responses = {}

for i, sentence in enumerate(benchie_sentences):
    out[i], responses[i] = get_chat_gpt_triples(sentence)

import pickle
from datetime import datetime

# Define the filename with timestamp

obj_to_dump = [out, responses]
filename = "benchie_chat_gpt_result_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pickle"
# Save the data to file using pickle
with open(filename, "wb") as f:
    pickle.dump(obj_to_dump, f)

"""
