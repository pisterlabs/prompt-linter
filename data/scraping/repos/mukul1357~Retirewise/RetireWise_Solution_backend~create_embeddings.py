from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
import json

def process_data(datas):
    output = []
    i = 0
    for data in datas:
        portfolios = data["portfolios"]
        prepend = ""
        if i // 2 == 0:
            prepend = "Low Risk Conservative Scheme"
        elif i // 2 == 1:
            prepend = "Moderate Risk Balanced Scheme"
        else:
            prepend = "High Risk Aggressive Scheme"
        for portfolio in portfolios:
            output.append(f"{prepend} {portfolio['name']}: {portfolio['description']}")
        i+=1
    return output

file_path = "data.json"
with open(file_path, 'r') as file:
    old_data = json.load(file)

persist_directory = 'db'
vectorstore = Chroma.from_texts(process_data(old_data), embedding=OpenAIEmbeddings(model_name="text-embedding-ada-002"), persist_directory=persist_directory)
vectorstore.persist()