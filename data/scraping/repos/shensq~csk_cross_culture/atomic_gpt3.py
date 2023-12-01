import os
import openai
import json
import pickle
from tqdm import tqdm
from api_key import api_key

openai.api_key = api_key

def get_response(text):
    try:
        response = openai.Completion.create(
          model="text-curie-001",
          prompt=text,
          temperature=0,
          max_tokens=32,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          logprobs = 5
        )
    except:
        print("a bad query for {}".format(text))
    return response


percents = [0.1,]
genders = ["male",'female']
langs = ['zh_CN','en_US']

for gender in genders:
    for lang in langs:
        for percent in percents:
            with open("data/processed_{}_{}_{}.json".format(percent, lang, gender),'r') as f:
                data = f.readlines()
                data = [json.loads(d) for d in data]
            results = []
            counter = 0
            num_samples = len(data)
            for sample in tqdm(data):
                text = sample["text"]
                response = get_response(text)
                results.append({"input":text, "response":response})
                counter += 1
                if counter > num_samples:
                    break
            with open("gpt3/{}_{}_{}.json".format(percent, lang, gender),'w') as f:
                json.dump(results, f)