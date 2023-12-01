import urllib.request
import openai
import csv

openai.organization = "org-"
openai.api_key = "sk-"
file_path = "clothing.csv"
data = []

with open(file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    for row in csv_reader:
        country = row["Country"]
        male = row["Men"]
        female = row["Women"]
        
        data.append({"Country": country, "Men": male, "Women": female})

for entry in data:
    filename_female = entry["Country"] + "-f.jpg"
    prompt_female = "Full view Foto of a 22-year-old woman from " + entry["Country"] + " dressed in traditional clothing like " + entry["Women"]  + ". Realistic, studio lights, posture is upright, vertical shot"
    response = openai.Image.create(prompt=prompt_female, n=2, size="1024x1024")
    urllib.request.urlretrieve(response["data"][0]["url"], filename_female)

    filename_male = entry["Country"] + "-m.jpg"
    prompt_male = "Full view Foto of a 22-year-old man from " + entry["Country"] + " dressed in traditional clothing like " + entry["Men"]  + ". Realistic, studio lights, posture is upright, vertical shot"
    response = openai.Image.create(prompt=filename_male, n=2, size="1024x1024")
    urllib.request.urlretrieve(response["data"][0]["url"], filename_male)
