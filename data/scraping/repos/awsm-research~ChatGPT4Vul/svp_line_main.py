
import pandas as pd
import pickle
import openai

df = pd.read_csv("../data/svp/gpt35_prediction.csv")
functions = df["func_before"].tolist()
functions_cut = []
line_count = []
for func in functions:
    # word-level 400 tokens would equal to more than 512 subword tokens
    f_cut = func.split(" ")[:400]
    f_cut = " ".join(f_cut)
    functions_cut.append(f_cut)
    line_count.append(len(f_cut.split("\n")))
print("total number of samples: ", len(functions_cut))

openai.api_key = "YOUR-KEY-HERE"

start = 0
for idx in range(start, len(functions_cut)):
    one_function = functions_cut[idx]
    loc = line_count[idx]
    sample_pred = [0 for _ in range(loc)]
    messages = [{"role": "user", "content": """The following C/C++ function/snippet is vulnerable. Predict which of the 10 lines are the most vulnerable-prone.
Return template:
Line 1: code...
Line 2: code...
Line n: code...

Generate code only without any explanation."""},
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
    
    with open(f"../response/svp_line_files/{model_name}/{idx}.pkl", "wb+") as f:
        pickle.dump(chat_completion, f)
        
    print(chat_completion.choices[0].message.content)
    print("-----------------------------------------")
print("done")