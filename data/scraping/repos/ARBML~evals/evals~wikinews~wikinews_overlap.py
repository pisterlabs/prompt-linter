import os
import sys
import json
import time
import openai
import requests 
import numpy as np
from tqdm import tqdm
from pyarabic.araby import strip_tashkeel
from torch.utils.data import Dataset, DataLoader

MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 256
DOMAIN = "art"

MAX_ATTEMPTS = 10

def retry_request(url, payload, headers):
  for i in range(MAX_ATTEMPTS):
    try:
      response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=90)
      return json.loads(response.content)
    except:
      print(f"> Sleeping for {2 ** i}")
      time.sleep(2 ** i) # exponential back off
  raise TimeoutError()


def read_file(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = fin.readlines()
    return data 

def read_json(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = json.load(fin)
    return data 

def write_file(path, data):
    with open(path, 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(data))

def write_json(path, data):
    with open(path, 'w', encoding="utf-8") as fout:
        json.dump(data, fout)

class WikiNews(Dataset):
    def __init__(self, dirpath, domain="sports") -> None:
        filename = f"WikiNews.{domain}.2-20.txt"
        self.data = read_file(os.path.join(dirpath, filename))
        self.set_instruction("Please diacritize the following Arabic sentence")

    def set_instruction(self, instruction):
        self.instruction = instruction

    def getdata(self, max_n=None):
        max_n = max_n or len(self.data)
        return self.data[:max_n]
    
    def stats(self):
        print(f"# Lines: {len(self)}")
        num_words = []
        for line in self.data:
            num_words += [len(line.split())]
        print(f"Mean: {np.mean(num_words)} | Stdev: {np.std(num_words)}")
        print(f"Max: {np.max(num_words)} | Min: {np.min(num_words)}")


    def __getitem__(self, index):
        line = strip_tashkeel(self.data[index])
        payload = {"role": "user", "content": f"{self.instruction}:\n{line}"}
        return payload
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    API_KEY = os.getenv("OPENAI_API_KEY")
    openai.organization = "org-6VIsbC1WgU4rx2bWplxNV7gP"
    openai.api_key = API_KEY

    if len(sys.argv) >= 2:
        DOMAIN = sys.argv[1]
        print(f"> Domain: {DOMAIN}")
        
    dirpath = f"/Users/bkhmsi/Desktop/WikiNews/{DOMAIN}-overlap"
    usage_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}_temp={TEMPERATURE}.2-20.usage.json")
    preds_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}_temp={TEMPERATURE}.2-20.pred")

    dataset = WikiNews(dirpath, domain=DOMAIN)

    instruction = "Please diacritize the following Arabic sentence"
    dataset.set_instruction(instruction)
    dataset.stats()

    max_gens = len(dataset)
    
    completions, usage = [], []
    if os.path.exists(preds_path):
        completions = read_file(preds_path)
        usage = read_json(usage_path)
    
    assert len(completions) == len(usage)
    start_idx = len(completions)
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {'Content-type': 'application/json', 'Accept': 'application/json', 'Authorization': f'Bearer {API_KEY}'}

    for index in tqdm(range(start_idx, max_gens)):

        payload = {"messages": [dataset[index]], "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "model": MODEL_NAME}
        response = retry_request(url, payload, headers)

        if "choices" in response:
            diac_text = response["choices"][0]["message"]["content"]
            completions += [diac_text.strip()]
            usage += [response["usage"]]
        else:
            print("> Error!")
            completions += ["Error"]
            usage += [{"Error": 0}]

        write_file(preds_path, completions)
        write_json(usage_path, usage)
    
# batch = [dict(zip(batch,t)) for t in zip(*batch.values())]
# response = openai.ChatCompletion.create(
#     model=MODEL_NAME,
#     messages=[dataset[index]],
#     max_tokens=MAX_TOKENS,
#     temperature=TEMPERATURE
# )

