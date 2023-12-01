from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv('.env')


def x(description, title, tags, transcript):
    #Init Anthropic
    anthropic = Anthropic()

    # Init prompt
    prompt = "Voici la transcription d'un TED Talk" + transcript + "\n" + "Voici le titre du TED Talk" + title + "\n" + "Voici la description du TED Talk" + description + "\n" + "Voici les tags du TED Talk" + tags + "\n" + "Je voudrais qu'à partir de ce texte tu determines une structure sémantique générique. Cette structure permettrait d'en faire un modèle pouvant être réutilisé pour écrire un nouveau talk selon la même structure. Elle devra etre décrite par une syntaxe à base d'accolades comme ceci : (ex: {Introduction {point1, point2, point3}, Corps {pointa, pointb, pointc}, Conclusion {pointx, pointy, pointz}}) Le format de ce shéma devrait etre assez précis et informatif pour que tu puisses le comprendre et l'utiliser si je te fournis des informations (thème, ton, source d'information) pour écrire un nouveau talk"

    # Call the API
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )

    print(completion.completion)
    return completion.completion

# Environment Variables
api_key = ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']


# Lecture des premières lignes du nouveau fichier pour comprendre sa structure
sample_data_new = pd.read_csv("test.csv", sep=";", nrows=2450)

# Appliquer la fonction x à chaque ligne du DataFrame
sample_data_new['structure'] = sample_data_new.apply(
    lambda row: x(row['description'], row['title'], row['tags'], row['transcript']), axis=1)

# Afficher les premières lignes du DataFrame modifié pour vérification
sample_data_new.head()


# Enregistrement du DataFrame modifié dans un nouveau fichier CSV
output_file_path = "test_modified.csv"
sample_data_new.to_csv(output_file_path, sep=";", index=False)

output_file_path
