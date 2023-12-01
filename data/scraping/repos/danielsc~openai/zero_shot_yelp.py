import pandas as pd
import numpy as np
import openai
import argparse
import mlflow
from langchain.prompts import PromptTemplate
from openai_llm import AzureOpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt
import tempfile
from util import find_first_number, calculate_metrics, save_yaml


def classify_yelp_batch(llm, 
                        texts,
                        prompt, 
                        batch_size=20):
    results = []
    input_prompt = PromptTemplate(input_variables=["text"], template=prompt)

    for i in tqdm(range(0, len(texts), batch_size)):
        prompts = [ input_prompt.format(text=text) for text in texts[i:i+batch_size]]
        batch_results = llm(prompts)
        batch_results = [find_first_number(item) for item in batch_results]
        results.extend(batch_results)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/1raw/yelp_10.csv")
    parser.add_argument("--prompts", default="./data/1prompts/prompts.csv")
    parser.add_argument("--prompt_number", default=1, type=int)
    parser.add_argument("--test_output", default="./data/1zero_shot/metrics.yaml")
    parser.add_argument("--prompt_column", default="text")
    parser.add_argument("--completion_column", default="stars")
    parser.add_argument("--aoai_endpoint", default="https://aoai.openai.azure.com/")
    parser.add_argument("--api_version", default="2022-06-01-preview")
    parser.add_argument("--deployment", default="text-davinci-002")
    # add temperture, max_tokens, top_p, frequency_penalty, presence_penalty
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=64, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--frequency_penalty", default=0, type=float)
    parser.add_argument("--presence_penalty", default=0, type=float)    
    args = parser.parse_args()

    mlflow.log_params({
        "data": args.data,
        "test_output": args.test_output,
        "prompt_column": args.prompt_column,
        "completion_column": args.completion_column,
        "aoai_endpoint": args.aoai_endpoint,
        "api_version": args.api_version,
        "deployment": args.deployment,
        "prompt_number": args.prompt_number,
        "prompts": args.prompts,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty
    })

    file = args.data
    print(f"processing {file}")
    df = pd.read_csv(file)

    df[args.completion_column] = df[args.completion_column].astype(str)
    
    # get prompt
    prompts = pd.read_csv(args.prompts)
    prompt = prompts.iloc[args.prompt_number]['prompt']
    print(f"prompt: {prompt}")
    mlflow.log_param("prompt", prompt[:1000])

    llm = AzureOpenAI(deployment=args.deployment,
                      api_base=args.aoai_endpoint,
                      temperature=args.temperature,
                      max_tokens=args.max_tokens,
                      top_p=args.top_p,
                      frequency_penalty=args.frequency_penalty,
                      presence_penalty=args.presence_penalty)

    results = classify_yelp_batch(llm=llm,
                                  texts=df[args.prompt_column].values, 
                                  prompt=prompt)

    df['prediction'] = results
    #print(df[['prediction','stars']])
    metrics = calculate_metrics(df[args.completion_column].values, results)

    bins = range(0, 7)
    _, temp_file_path = tempfile.mkstemp(suffix=".png")
    plt.hist([results, df[args.completion_column].values], bins, label=['predictions', 'stars'])
    plt.legend(loc='upper center')
    plt.savefig(temp_file_path)


    print("File path: " + temp_file_path)
    mlflow.log_artifact(temp_file_path, "hist_predictions_vs_truth.png")
    mlflow.log_metrics(metrics)
    print(metrics)

    save_yaml({
        "data": args.data,
        "test_output": args.test_output,
        "prompt_column": args.prompt_column,
        "prompt": prompt,
        "completion_column": args.completion_column,
        "aoai_endpoint": args.aoai_endpoint,
        "api_version": args.api_version,
        "completion_deployment": args.deployment,
        "metrics": metrics
    }, args.test_output)





