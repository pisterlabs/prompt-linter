from langchain.prompts import PromptTemplate
import evaluate
import pandas as pd
from openai_llm import AzureOpenAI
from datasets import load_dataset
import numpy as np
import mlflow

np.random.seed(42)

dataset = load_dataset("opus100", "en-fr", split="test")

# Get a sample of 3 sentences
indexes = np.random.choice(len(dataset), 100, replace=False)

translations = dataset.select(indexes)['translation']

llm = AzureOpenAI(deployment="text-davinci-002", temperature=0.5, max_tokens=100)
google_bleu = evaluate.load("google_bleu")

metrics = []
prefixes = ["Attached is a sentence translated into two languages",
    "See below for a sentence in two different languages",
    "The sentence in two languages can be found below",
    "Find the sentence in two languages below",
    "Below is the sentence in two languages"]
    
references = [translation["fr"] for translation in translations]
for prefix in prefixes:
    print(prefix)
    input_prompt = PromptTemplate(input_variables=["prefix", "english"], template="{prefix}\n\nEnglish: {english}\nFrench: ")

    prompts = []
    for translation in translations:
        prompts.append(input_prompt.format(prefix=prefix, english=translation['en']))
    
    predictions = llm(prompts)

    bleu_metric = google_bleu.compute(predictions=predictions, 
                               references=references)

    print(bleu_metric)
    metrics.append({"prefix":prefix,**bleu_metric})

print(pd.DataFrame(metrics))