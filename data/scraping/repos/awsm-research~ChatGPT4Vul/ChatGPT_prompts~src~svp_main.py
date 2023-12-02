
import pandas as pd
import pickle
import openai

df = pd.read_csv("../../data/svp/test.csv")

functions = df["func_before"].tolist()
functions_cut = []

for func in functions:
    # word-level 400 tokens would equal to more than 512 subword tokens
    f_cut = func.split(" ")[:400]
    f_cut = " ".join(f_cut)
    functions_cut.append(f_cut)
    
print("total number of samples: ", len(functions_cut))

openai.api_key = "YOUR-KEY-HERE"

start = 0
for idx in range(start, len(functions_cut)):
    one_function = functions_cut[idx]
    messages = [{"role": "user", "content": 'Predict whether C/C++ function/snippet below is vulnerable. Strictly return 1 for a vulnerable function or 0 for a non-vulnerable function without any other text.'},
    {"role": "user",
    "content": one_function}]
    print("ChatGPT start...")
    model_name = "gpt-3.5-turbo"
    # create a chat completion
    try:
        chat_completion = openai.ChatCompletion.create(model=model_name, messages=messages)
    except:
        print("ChatGPT server error!")
        print(f"terminated at index {idx} ...")
        print("try to restart...")
        chat_completion = openai.ChatCompletion.create(model=model_name, messages=messages)
    print("ChatGPT completed...")
    with open(f"../response/svp_files/{model_name}/{idx}.pkl", "wb+") as f:
        pickle.dump(chat_completion, f)
    
    if '0' not in chat_completion.choices[0].message.content and '1' not in chat_completion.choices[0].message.content:
        print("error no integer prediction returned", "\n", chat_completion.choices[0].message.content)
        print(f"terminated at index {idx} ...")
        chat_completion.choices[0].message.content = "0"
        with open(f"../response/svp_files/{model_name}/{idx}.pkl", "wb+") as f:
            pickle.dump(chat_completion, f)
        exit()
    else:
        print(chat_completion.choices[0].message.content)

print("done")