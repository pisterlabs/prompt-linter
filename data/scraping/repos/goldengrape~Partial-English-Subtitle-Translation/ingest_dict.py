# 理论上可行，
# 但实际上ecdict.csv有770612行
# 所以成本和时间都太高了

import pandas as pd
import openai
import os 

openai.api_key = os.environ.get("OPENAI_API_KEY")

def read_csv_file(file_path):
    return pd.read_csv(file_path,low_memory=False)

def get_word_embeddings(word):
    response = openai.Embedding.create(
        input=word,
        engine="text-similarity-davinci-001"
    )
    return response["data"][0]["embedding"]

def create_dataframe_with_embeddings_and_additional_info(data):
    embeddings = [get_word_embeddings(word) for word in data['word']]
    new_data = data[['collins', 'oxford', 'tag', 'frq']]
    new_data['embeddings'] = embeddings
    return new_data

def save_dataframe_to_pickle(dataframe, output_path):
    dataframe.to_pickle(output_path)

def main():
    file_path = "ecdict.csv"
    output_path = "dict.pkl"
    data = read_csv_file(file_path)
    new_dataframe = create_dataframe_with_embeddings_and_additional_info(data)
    save_dataframe_to_pickle(new_dataframe, output_path)

if __name__ == "__main__":
    
    main()
