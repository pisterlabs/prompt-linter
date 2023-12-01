
import pandas as pd
import pickle
import os
import openai
import time

df = pd.read_csv("../../data/svc_severity/test.csv")
functions = df["func_before"].tolist()
functions_cut = []

print("total samples: ", len(df))

for func in functions:
    # word-level 400 tokens would equal to more than 512 subword tokens
    f_cut = func.split(" ")[:400]
    f_cut = " ".join(f_cut)
    functions_cut.append(f_cut)

openai.api_key = "YOUR-KEY-HERE"
models = openai.Model.list()

MODEL_NAME = "gpt-3.5-turbo"

start = 0
for idx in range(start, len(functions_cut)):
    one_function = functions_cut[idx]
    
    messages = [{"role": "user", "content": "The C/C++ function/snippet below is confirmed vulnerable, identify the CWE-ID of this vulnerability. The potential CWE-IDs are ['CWE-264', 'CWE-119', 'CWE-20', 'CWE-190', 'CWE-200', 'CWE-415', 'CWE-787', 'CWE-416', 'CWE-476', 'CWE-399', 'CWE-284', 'CWE-189', 'CWE-404', 'CWE-362', 'CWE-352', 'CWE-772', 'CWE-494', 'CWE-125', 'CWE-611', 'CWE-254', 'CWE-704', 'CWE-18', 'CWE-17', 'CWE-79', 'CWE-732', 'CWE-129', 'CWE-754', 'CWE-22', 'CWE-310', 'CWE-835', 'CWE-134', 'CWE-834', 'CWE-59', 'CWE-617', 'CWE-19', 'CWE-347', 'CWE-862', 'CWE-285', 'CWE-674', 'CWE-682', 'CWE-120', 'CWE-287', 'CWE-269', 'CWE-358', 'CWE-77', 'CWE-311', 'CWE-665', 'CWE-400', 'CWE-369', 'CWE-94']. The potential severity score is a float between 0 to 10. Strictly return (1) one of the CWE-IDs from the list and (2) a severity score estimation, without any other text."},
    {"role": "user",
    "content": one_function}]
    
    print("ChatGPT start...")
    # create a chat completion
    chat_completion = openai.ChatCompletion.create(model=MODEL_NAME, messages=messages)
    
    print("ChatGPT completed...")

    with open(f"../response/svc_sev_files/{MODEL_NAME}/{idx}.pkl", "wb+") as f:
        pickle.dump(chat_completion, f)
        if "CWE-" not in chat_completion.choices[0].message.content:
            print("error", "\n", chat_completion.choices[0].message.content)
            print(f"terminated at index {idx} ...")
            exit()
        else:
            print(chat_completion.choices[0].message.content)
    time.sleep(2)
