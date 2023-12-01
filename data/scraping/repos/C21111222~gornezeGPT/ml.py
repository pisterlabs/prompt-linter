import pandas as pd
import base64
import io
import openai
import ml_model as mlm

def prompt(txt):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user", "content": txt}
    ])
    return completion.choices[0].message.content

def parse_content(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        res = prompt(f"Here is the head of my dataset : {df.head()}. Is it more likely a classification, a multi-classification or a regression problem ? And what column am I more likely wanting to predict ?")
    else:
        print("no")
    return df,res

def train_model(df,problem,variable):
    df = pd.read_json(df)
    if problem == "Classification":
        result,question,fig = mlm.train(df,variable)
    elif problem == "Multi-Classification":
        result,question,fig = mlm.train(df,variable,multi=True)
    elif problem == "Regression":
        result,question,fig = mlm.train(df,variable,reg=True)
    advices = prompt(question)
    return result,advices,fig


