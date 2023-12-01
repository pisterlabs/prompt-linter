# Package 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
## import re

## progress bar
from tqdm import tqdm
from random import sample
## from sklearn.cluster import KMeans
import random
import os
import pickle
import openai
from scipy.stats import entropy
import sys
import time
##-----------------------------------------------------------

## Load the GSM8K MATH Datasets
sys.path.append("datasets/")
from GSM8k_Prompt import *

##-----------------------------------------------------------

## Load the GPT3 GPT4 Text-Davinci-003 models
sys.path.append("models/GPT")
from GPT_API_3 import *
from GPT_API_4 import *
from Text_Davinci_003 import * 


OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

def Davinci_openai_short(prompt):
    response = openai.Completion.create(
    model="text-davinci-002",prompt = prompt, temperature = 1,
    max_tokens=16, top_p=1, frequency_penalty=0,
    presence_penalty=0,logprobs = 5)
    return response

def Davinci_openai_long(prompt):
    response = openai.Completion.create(
    model="text-davinci-002", prompt = prompt,  temperature= 1,
    max_tokens=2048, top_p=1, frequency_penalty=0,
    presence_penalty=0,logprobs = 5)
    return response

# -----------------------------------------------------------
# function for data processing and plot
def single_test_processing(dic_result):
    # data processing 
    n_round = 30
    df = pd.DataFrame(columns=["token", 'logprob'])
    for i in tqdm(range(n_round)):
        result_i = dic_result[i]["choices"]
        if '{' in  result_i[0]["logprobs"]["tokens"]:
            index = result_i[0]["logprobs"]["tokens"].index('{')
        else:
            index = result_i[0]["logprobs"]["tokens"].index(' {')
        sample_token = result_i[0]["logprobs"]["tokens"][index+1]
        sample_token_logprob = result_i[0]["logprobs"]["token_logprobs"][index+1]
        df.loc[len(df)] = [sample_token, sample_token_logprob]  
        log_list = result_i[0]["logprobs"]["top_logprobs"][index+1]
        for key in list(log_list.keys()):
            df.loc[len(df)] = [key, log_list[key]]
        df["token"] = [i.replace(" ","") for i in df["token"]]

    df_result = pd.DataFrame(columns=["token", 'logprob'])
    for token in np.unique(df["token"]):
        df_result.loc[len(df_result)] = [token, np.mean(df[df["token"] == token]["logprob"])]
    df_result["prob"] = np.exp(df_result["logprob"])
    df_result["prob"] = df_result["prob"]/np.sum(df_result["prob"])
    df_result["logprob"] = np.log(df_result["prob"])
    df_result = df_result.sort_values(by = "prob", ascending = False).reset_index(drop=True)

    df_result["cumprob"] = np.cumsum(df_result["prob"])

    df_result_majority = df_result[df_result["cumprob"] < 1] 

    df_result_majority = df_result_majority[[i.replace(' ','').isnumeric() for i in df_result_majority["token"]]].reset_index(drop=True)

    df_result_majority["prob"] = np.exp(df_result_majority["logprob"])
    df_result_majority["prob"] = df_result_majority["prob"]/np.sum(df_result_majority["prob"])

    df_result_majority["cumprob"] = np.cumsum(df_result_majority["prob"])
    return df_result_majority

# data process function
def single_test_plot(df_result_majority):
    # Plot
    fig = plt.figure(figsize=(4, 3))
    plt.subplot(2, 1, 1)
    plt.bar(df_result_majority["token"], df_result_majority["prob"])
    plt.xlabel('Categories')
    plt.ylabel('Probability')
    plt.title('Categorical Probability Distribution')

    # Plot the corresponding cumulative probability distribution
    plt.subplot(2, 1, 2)
    plt.plot(df_result_majority["token"], df_result_majority["cumprob"], marker='o', linestyle='-', color='r')
    plt.xlabel('Categories')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Probability Distribution')

    # Adjust the layout for better readability
    plt.tight_layout()

    # Save the plot to a file (optional)
    plt.savefig('plot.png')

    # Show the plot
    plt.show()

    # Entropy Calculation 
    H = entropy(df_result_majority['prob'])
    print("Entropy: " + str(H))
    return H
# -----------------------------------------------------------

def large_scale_data_processing(dic_result_collection,GSM8K_test_df_min):
    # data processing
    dic_result_df = {}
    df_Q_summary = pd.DataFrame(columns=['question','answer_clean', 
                                'test_answer', 'answer_list', 'prob_list', 'entropy', 'index_distance'])
    n_round = 5
    for i in tqdm(range(len(GSM8K_test_df_min))):
        dic_result_i = dic_result_collection[i]
        df = pd.DataFrame(columns=["token", 'logprob'])
        index_distence = 0
        for j in range(n_round):
            result_i = dic_result_i[j]["choices"]
            if '{' in result_i[0]["logprobs"]["tokens"]:
                index = result_i[0]["logprobs"]["tokens"].index('{')
            else:
                index = result_i[0]["logprobs"]["tokens"].index(' {')

            if '}' in result_i[0]["logprobs"]["tokens"]:
                index_next = result_i[0]["logprobs"]["tokens"].index('}')
            elif ' }' in result_i[0]["logprobs"]["tokens"]:
                index_next = result_i[0]["logprobs"]["tokens"].index(' }')
            else:
                index_next = 100

            index_distence += (index_next - index)

            sample_token = result_i[0]["logprobs"]["tokens"][index+1]
            sample_token_logprob = result_i[0]["logprobs"]["token_logprobs"][index+1]
            
            df.loc[len(df)] = [sample_token, sample_token_logprob]

            log_list = result_i[0]["logprobs"]["top_logprobs"][index+1]
            for key in list(log_list.keys()):
                df.loc[len(df)] = [key, log_list[key]]
            df["token"] = [i.replace(" ","") for i in df["token"]]

        df_result = pd.DataFrame(columns=["token", 'logprob'])
        for token in np.unique(df["token"]):
            df_result.loc[len(df_result)] = [token, np.mean(df[df["token"] == token]["logprob"])]

        df_result["prob"] = np.exp(df_result["logprob"])
        df_result["prob"] = df_result["prob"]/np.sum(df_result["prob"])
        df_result["logprob"] = np.log(df_result["prob"])
        df_result = df_result.sort_values(by = "prob", ascending = False).reset_index(drop=True)

        df_result["cumprob"] = np.cumsum(df_result["prob"])

        df_result_majority = df_result[df_result["cumprob"] < 1] 

        df_result_majority = df_result_majority[[i.replace(' ','').isnumeric() for i in df_result_majority["token"]]].reset_index(drop=True)

        df_result_majority["prob"] = np.exp(df_result_majority["logprob"])
        df_result_majority["prob"] = df_result_majority["prob"]/np.sum(df_result_majority["prob"])

        df_result_majority["cumprob"] = np.cumsum(df_result_majority["prob"])

        df_Q_summary.loc[len(df_Q_summary)] = [GSM8K_test_df_min['question'][i], 
                                                int(GSM8K_test_df_min['answer_clean'][i]),
                                                int(df_result_majority["token"][0]),
                                                [int(i) for i in df_result_majority["token"]],
                                                list(df_result_majority["prob"]),
                                                entropy(list(df_result_majority["prob"])),
                                                index_distence/n_round]
    df_Q_summary["correct"] = list(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])
    print("Accuracy: " + str(np.round(np.sum(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])/len(df_Q_summary)*100,2)) + "%")
    return df_Q_summary

def summary_processing_plot(df_Q_summary):
    # entropy calculation
    df_Q_summary = df_Q_summary[df_Q_summary["index_distance"] == 2]
    print(len(df_Q_summary))
    print("Accuracy: " + str(np.round(np.sum(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])/len(df_Q_summary)*100,2)) + "%")

    df_Q_summary_true = df_Q_summary[df_Q_summary["answer_clean"] == df_Q_summary["test_answer"]].reset_index(drop=True)

    print("Entropy_True " + str(np.mean(df_Q_summary_true["entropy"])))

    df_Q_summary_false = df_Q_summary[df_Q_summary["answer_clean"] != df_Q_summary["test_answer"]].reset_index(drop=True)

    print("Entropy_False " + str(np.mean(df_Q_summary_false["entropy"])))

    # plot
    fig, axs = plt.subplots(2, 1, figsize=(4, 3))
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    sns.set(style="darkgrid")
    sns.histplot(data=df_Q_summary_false, x="entropy", color="red", label="false", kde=True, ax=axs[0], bins=20)

    sns.histplot(data=df_Q_summary_true, x="entropy", color="skyblue", label="correct", kde=True, ax=axs[1], bins=20)

    plt.setp(axs, xlim = (0,max(df_Q_summary_false['entropy'])))
    plt.legend() 
    plt.show()
# -----------------------------------------------------------

def large_scale_data_processing_step(dic_result_collection, GSM8K_test_df_min, step_3_index_list):
    # data processing
    dic_result_df = {}
    df_Q_summary = pd.DataFrame(columns=['question','answer_clean', 
                                'test_answer', 'answer_list', 'prob_list', 'entropy', 'index_distance'])
    n_round = 5

    pbar = tqdm(total=len(step_3_index_list))

    for _ , i in enumerate(step_3_index_list):
        dic_result_i = dic_result_collection[i]
        df = pd.DataFrame(columns=["token", 'logprob'])
        index_distence = 0
        for j in range(n_round):
            result_i = dic_result_i[j]["choices"]
            if '{' in result_i[0]["logprobs"]["tokens"]:
                index = result_i[0]["logprobs"]["tokens"].index('{')
            else:
                index = result_i[0]["logprobs"]["tokens"].index(' {')

            if '}' in result_i[0]["logprobs"]["tokens"]:
                index_next = result_i[0]["logprobs"]["tokens"].index('}')
            elif ' }' in result_i[0]["logprobs"]["tokens"]:
                index_next = result_i[0]["logprobs"]["tokens"].index(' }')
            else:
                index_next = 100

            index_distence += (index_next - index)
            
            if index+1 >= len(result_i[0]["logprobs"]["tokens"]):
                continue

            sample_token = result_i[0]["logprobs"]["tokens"][index+1]
            sample_token_logprob = result_i[0]["logprobs"]["token_logprobs"][index+1]
            
            df.loc[len(df)] = [sample_token, sample_token_logprob]

            log_list = result_i[0]["logprobs"]["top_logprobs"][index+1]
            for key in list(log_list.keys()):
                df.loc[len(df)] = [key, log_list[key]]
            df["token"] = [i.replace(" ","") for i in df["token"]]
        pbar.update(1)
        df_result = pd.DataFrame(columns=["token", 'logprob'])
        for token in np.unique(df["token"]):
            df_result.loc[len(df_result)] = [token, np.mean(df[df["token"] == token]["logprob"])]
        df_result["token"] = [i.replace(" ","") for i in df_result["token"]]
        df_result["prob"] = np.exp(df_result["logprob"])
        df_result["prob"] = df_result["prob"]/np.sum(df_result["prob"])
        df_result["logprob"] = np.log(df_result["prob"])
        df_result = df_result.sort_values(by = "prob", ascending = False).reset_index(drop=True)

        df_result["cumprob"] = np.cumsum(df_result["prob"])

        df_result_majority = df_result[df_result["cumprob"] < 1] 

        df_result_majority = df_result_majority[[i.replace(' ','').isnumeric() for i in df_result_majority["token"]]].reset_index(drop=True)

        df_result_majority["prob"] = np.exp(df_result_majority["logprob"])
        df_result_majority["prob"] = df_result_majority["prob"]/np.sum(df_result_majority["prob"])

        df_result_majority["cumprob"] = np.cumsum(df_result_majority["prob"])

        df_Q_summary.loc[len(df_Q_summary)] = [GSM8K_test_df_min['question'][i], 
                                                int(GSM8K_test_df_min['answer_clean'][i]),
                                                int(df_result_majority["token"][0]),
                                                [int(i) for i in df_result_majority["token"]],
                                                list(df_result_majority["prob"]),
                                                entropy(list(df_result_majority["prob"])),
                                                index_distence/n_round]
    pbar.close()
    df_Q_summary["correct"] = list(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])
    print("Accuracy: " + str(np.round(np.sum(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])/len(df_Q_summary)*100,2)) + "%")
    return df_Q_summary