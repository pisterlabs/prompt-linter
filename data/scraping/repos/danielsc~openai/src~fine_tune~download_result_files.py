from time import sleep
import requests
import os, argparse
import openai, yaml
import mlflow, time
import pandas as pd
import io
from fine_tune import load_api_key, load_yaml, get_fine_tune 

def log_metrics(fine_tune, metric_logged):
  if "result_files" in fine_tune:
    result_file = fine_tune["result_files"][0]
    response = openai.File.download(id=result_file['id'])
    f = io.BytesIO(response)
    df = pd.read_csv(f)

    # step,elapsed_tokens,elapsed_examples,training_loss,training_sequence_accuracy,training_token_accuracy,validation_loss,validation_sequence_accuracy,validation_token_accuracy
    for col in df.columns[1:]: 
      values = df[['step',col]].dropna() 
      # drop all rows with -1 as a value
      values = values[values[col] != -1]
      for i, row in values.iterrows():
        if i >= metric_logged:
          #mlflow.log_metric(key=col, value=row[col], step=int(row.step)) 
          print(dict(key=col, value=row[col], step=int(row.step)))
    metric_logged = len(df)
  return metric_logged

def download_fine_tune_result_files(url, api_key, fine_tune_id, args):
  print("downloading result file for fine tune", fine_tune_id)
  openai.api_version = args.api_version
  openai.api_key = api_key
  openai.api_base = url
  openai.api_type = "azure"
  print("fine_tune_id: ", fine_tune_id)
  print("api_version: ", args.api_version)
  print("api_base: ", url)

  metric_logged = 0
  event_printed = 0
  while True:
    fine_tune = openai.FineTune.retrieve(id=fine_tune_id)

    for event in fine_tune['events'][event_printed:]:
      print()
      print(event)
      event_printed += 1
    
    metric_logged = log_metrics(fine_tune, metric_logged)

    print(".", end="", flush=True)

    if fine_tune["status"] in ["succeeded", "failed", "canceled"]:
      break

    time.sleep(10)

  return 

def main():
  parser = argparse.ArgumentParser()
  # add optional argument fine_tune_id
  parser.add_argument("--fine_tune_id", type=str, default=None)
  parser.add_argument("--fine_tune_metadata", default="./data/4fine_tune")
  parser.add_argument("--aoai_resource_name", default="aoai")
  parser.add_argument("--api_version", default="2022-06-01-preview")
  args = parser.parse_args()

  api_key = load_api_key()
  if not args.fine_tune_id:
    args.fine_tune_id = load_yaml(f"{args.fine_tune_metadata}/MLArtifact.yaml")["id"]

  url = f"https://{args.aoai_resource_name}.openai.azure.com"

  data = get_fine_tune(url + "/openai/fine-tunes", api_key, args.fine_tune_id, args)
  print(data)

  download_fine_tune_result_files(url, api_key, args.fine_tune_id, args)

if __name__ == "__main__":
  main()
