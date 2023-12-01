import openai
import os 
import pandas as pd
import numpy as np

from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

openai.api_key = "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"

def embed_skills(path="info_df.csv",embeddings_path="embeddings.csv"):
    superpowers_df = pd.read_csv(path)
    df = pd.DataFrame(superpowers_df.columns[1:6].transpose(), columns=['Intelligence'])
    df['Embedding'] = df['Intelligence'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    df.to_csv("embeddings_Skills.csv")
    return df 

def embed_powers(path="info_df.csv",embeddings_path="embeddings.csv"):
    superpowers_df = pd.read_csv(path)
    df = pd.DataFrame(superpowers_df.columns[8:].transpose(), columns=['Agility'])
    df['Embedding'] = df['Agility'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    df.to_csv(embeddings_path)
    return df 

def buscarHabilidades(busqueda, datos, n_resultados=2):
    busqueda_embed = get_embedding(busqueda, engine="text-embedding-ada-002")
    datos["Similitud"] = datos["Embedding"].apply(lambda x: cosine_similarity(x, busqueda_embed))
    datos = datos.sort_values("Similitud", ascending=False)
    return datos.iloc[:n_resultados][["Intelligence", "Similitud"]]

def buscarPoder(busqueda, datos, n_resultados=10):
    busqueda_embed = get_embedding(busqueda, engine="text-embedding-ada-002")
    datos["Similitud"] = datos["Embedding"].apply(lambda x: cosine_similarity(x, busqueda_embed))
    datos = datos.sort_values("Similitud", ascending=False)
    return datos.iloc[:n_resultados][["Agility", "Similitud"]]

def AgregarPoder(superpoder):
    resultado = buscarPoder(superpoder, powers_emb, n_resultados=1)
    if resultado.iloc[0]["Similitud"] < 0.85:
        print("No se encontro el superpoder")
        AddSuperpower()
    else:
        skills = buscarHabilidades(resultado.iloc[0]["Agility"], skills_emb, n_resultados=5)
        n = 0
        skills = skills.sort_values("Similitud", ascending=False)
        for i in range(len(skills)):
            if(skills.iloc[i]["Similitud"]>0.85):
                n = n+1
        if n == 0:
            n = 1
        df = pd.read_csv("./df_filtrado.csv")
        heroes_with_skill = df[df[resultado.iloc[0]["Agility"]]==True][['Name', skills[0:n]["Intelligence"].to_string(index=False).split(",")[0]]]
        # print(f"heros with this skill: {len(heroes_with_skill)}\n{heroes_with_skill}")
        # Calculate extra points and damage points
        skill_name = skills[0:n]["Intelligence"].to_string(index=False).split(",")[0]
        skill_mean = heroes_with_skill[skill_name].mean()
        extra_points = np.array([skill_name, round(skill_mean/10)])
        damage_points = np.array([resultado.iloc[0]["Agility"], round((skill_mean/5) * .75)])
        # print(f"Extra points: {extra_points}")
        # print(f"Damage points: {damage_points}")
    return extra_points, damage_points

Player_counter = 0
n_skills = 6
i = 0
skill_set = np.array([['Intelligence', 'Strength', 'Speed', 'Durability', 'Power', 'Combat'],[0,0,0,0,0,0]])

def AddPlayer():
    PlayerName = input(f"Nombre del Jugador {Player_counter+1}: ")
    print("Da el puntaje a cada habilidad")
    AddPoints(i)
    return PlayerName, skill_set[1][0:n_skills]

def AddPoints(i, max_points=300):
    for i in range(n_skills):
        skill_set[1][i] = input(f"Puntaje para {skill_set[0][i]}") 
        points_left = max_points - int(skill_set[1][i])
        if  points_left < 0:
            print("no tienes suficientes puntos")
            AddPoints(i, max_points)
        if int(skill_set[1][i]) > 100 or int(skill_set[1][i]) < 0:
            print("No puedes tener mas de 100 puntos en una habilidad")
            AddPoints(i, max_points)
        else:
            max_points = points_left
            print(f"Te quedan {max_points} puntos")

def SumExtraPoints(extra_points, points):
    for i in range(len(skill_set[0])):
        if skill_set[0][i] == extra_points[0]:
            points[i] = int(points[i]) + int(extra_points[1])
    return points

def AddSuperpower():
    Superpower = input("Agregar superpoder: ")
    extra_points, damage_points = AgregarPoder(Superpower)
    print("Extra points: ", extra_points)
    print("Damage points: ", damage_points)
    Player_stats = SumExtraPoints(extra_points, points)
    return Player_stats, extra_points, damage_points

def AddPowers():
    extra_points_array = []
    damage_points_array = []
    for i in range(3):
        ps,ep,dp  = AddSuperpower()
        extra_points_array.append(ep)
        damage_points_array.append(dp)
    return ps, extra_points_array, damage_points_array

print("Cargando archivos...")
if(os.path.exists("./embeddings_Powers.csv")):
    powers_emb = pd.read_csv("./embeddings_Powers.csv")
    powers_emb = pd.DataFrame(powers_emb.iloc[0:][["Agility", "Embedding"]])
    print("El archivo embeddings_Powers.csv ya existe")
else:
    print("creating embeddings...")
    powers_emb = embed_powers("./df_filtrado.csv", "./embeddings_Powers.csv")
    print("archivo embeddings_Powers.csv creado")

if(os.path.exists("./embeddings_Skills.csv")):
    skills_emb = pd.read_csv("./embeddings_Skills.csv")
    skill_emb = pd.DataFrame(skills_emb.iloc[0:][["Intelligence", "Embedding"]])
    print("El archivo embeddings_Skills.csv ya existe")
else:
    print("creating embeddings...")
    skills_emb = embed_skills("./df_filtrado.csv", "./embeddings_Skills.csv")
    print("archivo embeddings_Skills.csv creado")

print("\nEl juego esta listo para comenzar!")

# Crear Personaje
print("\nCREA TU PERSONAJE:\n")
name, points = AddPlayer()
print("Nombre del Personaje: ", name)
print("Puntos: ", points)

# Agregar superpoder
print("Agrega un superpoder a tu personaje:")
Player_stats, epa, dpa = AddPowers()
print("Stats: ", Player_stats)
print("Extra points: ", epa)
print("Damage points: ", dpa)



