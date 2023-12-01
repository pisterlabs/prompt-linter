# use gpt-3 to create user descriptions for each disease
# input: set of diseases from the scraped Mayo disease to symptoms mapping

import os
import openai
import pickle
import pandas as pd
openai.api_key = os.getenv("OPENAI_API_KEY")

file = open("mayo_symptoms_embeddings.plk",'rb')
df= pickle.load(file)

usersym=[]
diseases=[]

for i,d in enumerate(df.disease.unique()):
    print(i, d)
    diseases.append(d)
    p="symptoms of "+d+" \n"
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=p,
        temperature=0.8,
        max_tokens=100,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )
    usersym.append(response["choices"][0]["text"])

output = pd.DataFrame(
    {'disease': diseases,
    'userquery': usersym
    })
output.to_pickle("./data/UserInputSymptoms.plk")


file = open("./data/UserInputSymptoms.plk",'rb')
