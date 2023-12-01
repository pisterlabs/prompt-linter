import openai
import requests
import json
import os

API_KEY = "***" 
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
FolderPath2 = "F:\Git\hackyeah3\plikitreningowe\\trening2.json"
datatraining = ""

openai.api_key = API_KEY

print ('models')
models = openai.FineTune.list()
for model in models['data']:
    print(model['id'])

print('jobs')
jobs = openai.FineTuningJob.list()
for job in jobs['data']:
    print(job['id'])

print(openai.FineTuningJob.retrieve("ftjob-Pvb7BPDNvaFT8yELgXWScjCp"))


#openai.FineTuningJob.cancel("ftjob-CrdhWyHCcuchwhvRp19FkX3H")
