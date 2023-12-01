import torch,time,numpy as np
from flask import jsonify
from transformers import BertTokenizerFast, BertForQuestionAnswering
from datetime import datetime
from . import mysql
import openai
import textwrap,re
import urllib, json,os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
key = os.environ.get("API_CHATGPT")
#print(key)
openai.api_key = key
device = "cuda" if torch.cuda.is_available() else "cpu" 
torch.device(device) 
modelCheckpoint = "indolem/indobert-base-uncased"

model_directory = os.path.abspath(os.path.join(__file__, "../../bert_model/model.bin")) 
model = torch.load(model_directory,map_location=torch.device('cpu'))
model.to(device)
tokenizer = BertTokenizerFast.from_pretrained(modelCheckpoint)
start_time = time.time()


def convertTuple1(tup):
    return ''.join([str(x) for x in tup])

# dari database

def bert_prediction_pilih_context(context,question):
  print(context)
  respon_model = []
  dictlogs = {}
  penyakit_terdeteksi = []
  print(context)
  print(question)
  gejala = []
  encodedData = tokenizer(question, context.lower(), padding=True, return_offsets_mapping=True, truncation="only_second", return_tensors="pt")
  offsetMapping = encodedData.pop("offset_mapping")[0]
  encodedData.to(device)
  model.to(device)
  jmltoken = len(encodedData["input_ids"][0])
  if jmltoken > 512:
    dictlogs.update({"status": False,"deskripsi":"maaf terjadi error di sistem kami tunggu beberapa saat untuk mencoba kembali"})
  model.eval() 
  with torch.no_grad(): # IMPORTANT! Do not computing gradient!
    outputs = model(encodedData["input_ids"], attention_mask=encodedData["attention_mask"]) # Feed forward. Without calculating loss.
  startLogits = outputs.start_logits[0].detach().cpu().numpy() # Getting logits, moving to CPU.
  endLogits = outputs.end_logits[0].detach().cpu().numpy() # Getting logits, moving to CPU.
  start_indexes = np.argsort(startLogits).tolist()
  end_indexes = np.argsort(endLogits).tolist()
  candidates = []
  for start_index in start_indexes:
    for end_index in end_indexes:
      if (
        start_index >= len(offsetMapping)
        or end_index >= len(offsetMapping)
        or offsetMapping[start_index] is None
        or offsetMapping[end_index] is None
      ):
        continue 
      if end_index < start_index or end_index - start_index + 1 > 25:
        continue
      if start_index <= end_index:
        start_char = offsetMapping[start_index][0]
        end_char = offsetMapping[end_index][1]
        candidates.append({
          "score": startLogits[start_index] + endLogits[end_index],
          "text": context[start_char: end_char]
        })
  candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
  # Menghitung skor terendah dan tertinggi dari kandidat
  min_score = min(candidate['score'] for candidate in candidates)
  max_score = max(candidate['score'] for candidate in candidates)
  print(min_score)
  print(max_score)
  candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]
  # Normalisasi semua skor kandidat
  for i, candidate in enumerate(candidates):
     # Normalisasi semua skor kandidate
    print(candidate['score'])
    candidate['normalized_score'] = (candidate['score'] - min_score) / (max_score - min_score)
    scoree = candidate['score']
    print(scoree)
    print(candidate['text'])
    if scoree<=0:
      prediction = 'false'
      normalized_scoree = str(candidate['normalized_score']) # convert float32 to string
      if prediction[0]=='true':
        dictlogs.update({"status": True,"deskripsi":prediction[1]})
      else:
        dictlogs.update({"status": False,"deskripsi":"maaf kami tidak berhasil mencari gejala yang sesuai dengan penyakit anda"})
    else:
      # rank = str(i+1) #convert number rank to string
      normalized_scoree = str(candidate['normalized_score']) # convert float32 to string\
      print(normalized_scoree)
      jawaban = candidate['text']
      status = True
      dictlogs.update({"status": status,"jawaban": jawaban,"score":normalized_scoree})
   
      respon_model.append(dictlogs)    
  print(respon_model)
  return jsonify(respon_model)

def bert_prediction(context,question):
  print(context)
  context = ', '.join(context)
  print(context)
  respon_model = []
  dictlogs = {}
  penyakit_terdeteksi = []
  print(question)
  gejala = []
  encodedData = tokenizer(question, context.lower(), padding=True, return_offsets_mapping=True, truncation="only_second", return_tensors="pt")
  offsetMapping = encodedData.pop("offset_mapping")[0]
  encodedData.to(device)
  model.to(device)
  jmltoken = len(encodedData["input_ids"][0])
  if jmltoken > 512:
    dictlogs.update({"status": False,"deskripsi":"maaf terjadi error di sistem kami tunggu beberapa saat untuk mencoba kembali"})
  model.eval() 
  with torch.no_grad(): # IMPORTANT! Do not computing gradient!
    outputs = model(encodedData["input_ids"], attention_mask=encodedData["attention_mask"]) # Feed forward. Without calculating loss.
  startLogits = outputs.start_logits[0].detach().cpu().numpy() # Getting logits, moving to CPU.
  endLogits = outputs.end_logits[0].detach().cpu().numpy() # Getting logits, moving to CPU.
  start_indexes = np.argsort(startLogits).tolist()
  end_indexes = np.argsort(endLogits).tolist()
  candidates = []
  for start_index in start_indexes:
    for end_index in end_indexes:
      if (
        start_index >= len(offsetMapping)
        or end_index >= len(offsetMapping)
        or offsetMapping[start_index] is None
        or offsetMapping[end_index] is None
      ):
        continue 
      if end_index < start_index or end_index - start_index + 1 > 25:
        continue
      if start_index <= end_index:
        start_char = offsetMapping[start_index][0]
        end_char = offsetMapping[end_index][1]
        candidates.append({
          "score": startLogits[start_index] + endLogits[end_index],
          "text": context[start_char: end_char]
        })
  candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
  # Menghitung skor terendah dan tertinggi dari kandidat
  min_score = min(candidate['score'] for candidate in candidates)
  max_score = max(candidate['score'] for candidate in candidates)
  print(min_score)
  print(max_score)
  candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:1]
  # Normalisasi semua skor kandidat
  for i, candidate in enumerate(candidates):
     # Normalisasi semua skor kandidate
    print(candidate['score'])
    candidate['normalized_score'] = (candidate['score'] - min_score) / (max_score - min_score)
    scoree = candidate['score']
    print(scoree)
    if scoree<=0:
      prediction = 'false'
      scoree = str(candidate['normalized_score']) # convert float32 to string
      if prediction[0]=='true':
        dictlogs.update({"status": True,"deskripsi":prediction[1]})
      else:
        dictlogs.update({"status": False,"deskripsi":"maaf kami tidak berhasil mencari gejala yang sesuai dengan penyakit anda"})
    else:
      # rank = str(i+1) #convert number rank to string
      scoree = str(candidate['normalized_score']) # convert float32 to string
      jawaban = candidate['text']
      status = True
      dictlogs.update({"status": status,"jawaban": jawaban,"score":scoree})
   
  respon_model.append(dictlogs)    
  print(respon_model)
  return jsonify(respon_model)

def random_question(ask):
  prompt = (ask)

  completions = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.5,
  )

  message = completions.choices[0].text
  print(message)
  
  return message