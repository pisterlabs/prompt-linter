import pandas as pd
import numpy as np
import openai
import argparse
import mlflow
import os

def load_api_key(keyname = "openai-key"):
  secret_value = os.getenv("OPENAI_API_KEY")
  if secret_value is None:
    try:
      from azureml.core.run import _OfflineRun
      from azureml.core import Run
      run = Run.get_context()
      secret_value = run.get_secret(name=keyname)
    except Exception as e:
      Exception("No OPENAI_API_KEY found in environment variables or keyvault", e)

  return secret_value

def split_dataframe(df, ratio):
    part_train = df.sample(frac = ratio)
    part_validation = df.drop(part_train.index)
    return part_train, part_validation

def featurize_data(text_list, embedding_deployment):
    # create empty np array
    embeddings = []
    
    for i, text in enumerate(text_list):
        response = openai.Embedding.create(
            input=text,
            engine=embedding_deployment)
        
        embeddings.append(response.data[0].embedding)       
        # log every 100 rows
        if i % 100 == 0:
            print(f"processed {i} rows") 
    return np.array(embeddings) 
    

def save_data(embedding, target, output_path):
    # save data
    # open file binary
    with open(output_path, 'wb') as f:
        np.savez(f, embedding=embedding, target=target)
    
    print(f"saved data to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./data/1raw/yelp.csv")
    parser.add_argument("--train_output", default="./data/2feature/train_ada.npz")
    parser.add_argument("--validation_output", default="./data/2feature/validation_ada.npz")
    parser.add_argument("--prompt_column", default="text")
    parser.add_argument("--completion_column", default="stars")
    parser.add_argument("--train_test_split", default=0.8, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--aoai_endpoint", default="https://aoai.openai.azure.com/")
    parser.add_argument("--api_version", default="2022-06-01-preview")
    parser.add_argument("--embedding_deployment", default="text-similarity-ada-001")
    args = parser.parse_args()

    mlflow.log_params({
        "raw_data": args.raw_data,
        "train_output": args.train_output,
        "validation_output": args.validation_output,
        "prompt_column": args.prompt_column,
        "completion_column": args.completion_column,
        "train_test_split": args.train_test_split,
        "seed": args.seed,
        "aoai_endpoint": args.aoai_endpoint,
        "api_version": args.api_version,
        "embedding_deployment": args.embedding_deployment
    })

    api_key = load_api_key()
    openai.api_version = args.api_version
    openai.api_key = api_key
    openai.api_base = args.aoai_endpoint
    openai.api_type = "azure"


    file = args.raw_data
    print(f"processing {file}")
    df = pd.read_csv(file)
    
    embedding = featurize_data(df[args.prompt_column], args.embedding_deployment)

    # split the embedding into train and test
    df["embedding"] = [x for x in embedding]
    train, validation = split_dataframe(df, args.train_test_split)

    # save the train and test data
    save_data(np.vstack(train.embedding), train[args.completion_column], args.train_output)
    save_data(np.vstack(validation.embedding), validation[args.completion_column], args.validation_output)



