import os
import json
import openai
import numpy as np
from tqdm import tqdm
from pyarabic.araby import strip_tashkeel
from torch.utils.data import Dataset, DataLoader

MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 1024
DOMAIN = "sports"

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
    def __init__(self, dirpath, diac=False) -> None:
        filename = "WikiNewsTruth.txt.diac" if diac else "WikiNewsTruth.txt"
        self.data_domain = {}
        self.cluster_by_domain(read_file(os.path.join(dirpath, filename)))
        self.domain_map = {'culture': 'ثقافة', 'health': 'صحة', 'politics': 'سياسة', 'science': 'علوم', 'sports': 'رياضة', 'art': 'فن', 'economics': 'اقتصاد'}
        self.set_domain("sports")
        self.set_instruction("Please diacritize the following Arabic sentence")

    def set_domain(self, domain):
        self.domain = self.domain_map[domain]

    def set_instruction(self, instruction):
        self.instruction = instruction

    def cluster_by_domain(self, data):
        for line in data:
            line = line.strip()
            if line[0] == "#":
                last_domain = strip_tashkeel(line[1:].strip())
                self.data_domain[last_domain] = []
            else:
                self.data_domain[last_domain] += [line]

    def getdata(self, max_n=None):
        max_n = max_n or len(self.data_domain[self.domain])
        return self.data_domain[self.domain][:max_n]
    
    def stats(self):
        print(f"# Lines: {len(self)}")
        num_words = []
        for line in self.data_domain[self.domain]:
            num_words += [len(line.split())]
        print(f"Mean: {np.mean(num_words)} | Stdev: {np.std(num_words)}")
        print(f"Max: {np.max(num_words)} | Min: {np.min(num_words)}")


    def __getitem__(self, index):
        line = self.data_domain[self.domain][index]
        payload = {"role": "user", "content": f"{self.instruction}:\n{line}"}
        return payload
    
    def __len__(self):
        return len(self.data_domain[self.domain])


if __name__ == "__main__":

    openai.organization = "org-6VIsbC1WgU4rx2bWplxNV7gP"
    openai.api_key = os.getenv("OPENAI_API_KEY")

    dirpath = "/Users/bkhmsi/Desktop/WikiNews"
    pred_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}_temp={TEMPERATURE}.pred")
    grnd_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}.grnd")
    usage_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}_temp={TEMPERATURE}.usage.json")

    dataset = WikiNews(dirpath)
    diac_dataset = WikiNews(dirpath, diac=True)

    dataset.set_domain(DOMAIN)
    diac_dataset.set_domain(DOMAIN)

    instruction = "Please diacritize the following Arabic sentence"
    dataset.set_instruction(instruction)

    max_gens = len(dataset)

    completions, usage = [], []
    if os.path.exists(pred_path):
        completions = read_file(pred_path)
        usage = read_json(usage_path)
    
    assert len(completions) == len(usage)
    start_idx = len(completions)
    for index in tqdm(range(start_idx, max_gens)):

        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[dataset[index]],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )

        diac_text = response["choices"][0]["message"]["content"]
        completions += [diac_text]
        write_file(pred_path, completions)

        usage += [response["usage"]]
        write_json(usage_path, usage)

    if not os.path.exists(grnd_path):
        write_file(grnd_path, diac_dataset.getdata(max_n=max_gens))
    command = f"python diac_eval.py -ofp {grnd_path} -tfp {pred_path}"

    print()
    print(command)
    print()

    os.system(command)
    
# batch = [dict(zip(batch,t)) for t in zip(*batch.values())]
# python diac_eval.py -ofp /Users/bkhmsi/Desktop/WikiNews/WikiNews.sports.0.fgrnd -tfp /Users/bkhmsi/Desktop/WikiNews/WikiNews.sports.0.fpred
# python diac_eval.py -ofp /Users/bkhmsi/Desktop/WikiNews/segments/WikiNews.fsports.grnd -tfp /Users/bkhmsi/Desktop/WikiNews/segments/WikiNews.sports.2-20.pred.combined