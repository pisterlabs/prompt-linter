import json
import openai
import csv
from sklearn.model_selection import train_test_split

with open("fine_tuning\key.txt", "r") as file:
    api_key = file.read().strip()  # .strip() est utilisé pour s'assurer qu'aucun espace ou saut de ligne n'est inclus

openai.api_key = api_key

# Charger les données du fichier CSV
with open("fine_tuning/sadness_tweets.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Convertir les données au format spécifié
training_data = []
for row in data:
    score = float(row['\ufeffScore'])
    classe = int(score * 10)
    prompt = row['Tweet'] + " ->"
    completion = str(classe / 10) + ".\n"
    training_data.append({"prompt": prompt, "completion": completion})

for entry in training_data:
    entry["prompt"] = "In the following tweet what is the rate of sadness over 10? here is the tweet: " + entry["prompt"]

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=42)

# Nom des fichiers de sortie
train_file_name = "fine_tuning/training_data_sadness_train.jsonl"
test_file_name = "fine_tuning/training_data_sadness_test.jsonl"

# Écrire les données d'entraînement au format JSONL
with open(train_file_name, "w") as output_file:
    for entry in train_data:
        json.dump(entry, output_file)
        output_file.write("\n")

# Écrire les données de test au format JSONL
with open(test_file_name, "w") as output_file:
    for entry in test_data:
        json.dump(entry, output_file)
        output_file.write("\n")


