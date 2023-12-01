
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv('.env')

def x(description, title, tags, transcript):
    #Init Anthropic
    anthropic = Anthropic()

    # Init prompt
    prompt = ("Voici la transcription d'un TED Talk" + transcript + "\n" + 
              "Voici le titre du TED Talk" + title + "\n" + 
              "Voici la description du TED Talk" + description + "\n" + 
              "Voici les tags du TED Talk" + tags + "\n" + 
              "Je voudrais qu'à partir de ce texte tu determines une structure sémantique générique. \
            Cette structure permettrait d'en faire un modèle pouvant être réutilisé pour écrire un nouveau talk selon la même structure. \
            Elle devra etre décrite uniquement par un plan au format suivant et ne contenir que des éléments de structure générique. \
            Par générique, j'entends que rien dans ce plan ne doit être en lien avec le sujet ou les contenus du talk. \
            Il ne doit rendre compte que de la structure logique du talk. VOici le format à utiliser :\n\
            Introduction \n\
                - point1 \n\
                - ... \n\
                - pointn \n\
            Corps du Talk \n\
                - point1 \n\
                - ... \n\
                - pointn \n\
            Conclusion \n\
                - point1 \n\
                - ... \n\
                - pointn \n\ \n\
            Le format de ce shéma devrait etre assez précis et informatif pour que tu puisses \
            le comprendre et l'utiliser si je te fournis des informations \
            (thème, ton, source d'information) pour écrire un nouveau talk")
 
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

output_file_path = "test_modified.csv"

# Écrire les en-têtes la première fois
sample_data_new.head(0).to_csv(output_file_path, sep=";", index=False)

# Parcourez chaque ligne du DataFrame
for index, row in sample_data_new.iterrows():
    # Appliquez la fonction x à la ligne actuelle
    row['structure'] = x(row['description'], row['title'], row['tags'], row['transcript'])
    
    # Écrivez cette ligne dans le fichier CSV
    row.to_frame().T.to_csv(output_file_path, mode="a", header=False, sep=";", index=False)

output_file_path



