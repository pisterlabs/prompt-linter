import time
import numpy as np
import openai
from pymongo import MongoClient
import spacy
import threading

# Configuración de la API de OpenAI (ChatGPT)
openai.api_key = "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"

# Cargamos el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

#Establecemos una conexion a la db
client = MongoClient("mongodb+srv://Ever:Galaxyj2prime123@nesocluster.gmtcgxm.mongodb.net/?retryWrites=true&w=majority")
db = client["CarrerasUniversitarias"]
carrers = db["Carreras"]
questions = db["Preguntas"]

def computeAverageVector(lista_palabras):
    vectores = [word.vector for word in nlp(" ".join(lista_palabras)) if word.has_vector]
    
    if not vectores:
        return None
    
    return np.mean(vectores, axis=0)

def cosineSimilarity(vector_promedio_1, vector_promedio_2):
    return np.dot(vector_promedio_1, vector_promedio_2) / (np.linalg.norm(vector_promedio_1) * np.linalg.norm(vector_promedio_2))

def calculateSimilarityPropositions(proposition_1, proposition_2):
    prop_1 = nlp(proposition_1)
    prop_2 = nlp(proposition_2)
    return prop_1.similarity(prop_2)

def compareWithCareers(carrers_scores, user_dictionary):
    print("Iniciando comparación del diccionario de usuario con las carreras")
    user_vector = computeUserVector(user_dictionary)

    for carrera in carrers.find():
        carrer_dictionary = carrera  # No necesitas cargarlo como JSON
        carrer_vector = computeCareerVector(carrer_dictionary)
        nombre_carrera = carrer_dictionary["nombre_carrera"]
        print(f"Obteniendo score de {nombre_carrera}")
        similarity_score = cosineSimilarity(user_vector, carrer_vector)
        carrers_scores.append((similarity_score, carrer_dictionary))

    print("Finalizando comparación del diccionario de usuario con las carreras")
    carrers_scores.sort(reverse=True)
    return carrers_scores

def computeUserVector(user_dictionary):
    user_attributes = [
        user_dictionary["perfil_estudiante"]["aptitudes_y_habilidades"],
        user_dictionary["perfil_estudiante"]["experiencias_culturales"],
        user_dictionary["perfil_estudiante"]["experiencias_previas"],
        user_dictionary["perfil_estudiante"]["intereses_personales"],
        user_dictionary["perfil_estudiante"]["motivaciones_personales"],
        user_dictionary["perfil_estudiante"]["rasgos_de_personalidad"],
        user_dictionary["perfil_estudiante"]["valores_personales"],
        user_dictionary["perfil_estudiante"]["hobbies"]
    ]
    user_attributes = [item for sublist in user_attributes for item in sublist if item]

    return computeAverageVector(user_attributes)

def computeCareerVector(career_dict):
    career_attributes = [
        career_dict["perfil_estudiante"]["aptitudes_y_habilidades"],
        career_dict["perfil_estudiante"]["experiencias_culturales"],
        career_dict["perfil_estudiante"]["experiencias_previas"],
        career_dict["perfil_estudiante"]["intereses_personales"],
        career_dict["perfil_estudiante"]["motivaciones_personales"],
        career_dict["perfil_estudiante"]["rasgos_de_personalidad"],
        career_dict["perfil_estudiante"]["valores_personales"],
        career_dict["perfil_estudiante"]["hobbies"]
    ]
    career_attributes = [item for sublist in career_attributes for item in sublist if item]

    return computeAverageVector(career_attributes)

# Variable global para rastrear la cantidad de solicitudes realizadas por minuto
requests_per_minute = 0

# Función para procesar las respuestas en segundo plano
def process_responses_in_background(user_dictionary, responses_to_send):
    global requests_per_minute

    for response in responses_to_send:
        # Verifica si se ha alcanzado el límite de velocidad y espera hasta que se restablezca.
        while requests_per_minute >= 3:
            time.sleep(10)  # Puedes ajustar el tiempo de espera según tus necesidades.

        # Utilizamos GPT-3 para analizar la respuesta y llenar el diccionario de proposiciones
        response_analysis = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Quiero que llenes las categorías, que no están llenas o que aún se pueden llenar, de esta estructura {user_dictionary} con las proposiciones, adjetivos, verbos, etc que se encuentren presentes en este enunciado : {response}. Retorname el diccionario modificado",
            max_tokens=150
        )
        user_dictionary = response_analysis

        # Incrementa el contador de solicitudes realizadas por minuto.
        requests_per_minute += 1

    print("Diccionario modificado en segundo plano")

# Función para ver las preguntas y recopilar respuestas de una etapa
def viewQuestions(user_dictionary, etapa_name):
    etapa = questions.find_one({etapa_name: {"$exists": True}})

    if etapa is not None:
        preguntas = etapa[etapa_name]
        respuestas = []

        t0 = time.time()
        iterator = 0

        for i, pregunta in enumerate(preguntas, start=1):
            respuesta = input(f"{i}. {pregunta} (Escribe tu respuesta aquí): ")
            respuestas.append(respuesta)
            tf = time.time()
            dt = tf - t0

            if (0 <= dt <= 60) and iterator < 3:
                iterator += 1
            if (0 <= dt <= 60) and iterator >= 3:
                # Procesar las respuestas en segundo plano y continuar con la siguiente pregunta
                responses_to_send = respuestas[:]
                print(f"Procesando respuestas '{responses_to_send}' en segundo plano")
                thread = threading.Thread(target=process_responses_in_background, args=(user_dictionary, responses_to_send,))
                thread.start()
                respuestas = []
                iterator = 0
                t0 = tf

        # Esperar a que todos los hilos hayan terminado antes de regresar el diccionario
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join()
                print("Esperando a que los hilos terminen")

    return user_dictionary

# Función para realizar el test
def makeTest(user_dictionary):
    etapas = ["etapa_1"]

    for etapa_name in etapas:
        print(f"Iniciando etapa: {etapa_name}")
        user_dictionary = viewQuestions(user_dictionary, etapa_name)
        print(f"Finalizada etapa: {etapa_name}")

    print("Todas las etapas han sido completadas")

    return user_dictionary
    
def VocationLabTest(user_dictionary):
    career_scores = compareWithCareers([], user_dictionary)

    # Imprimir las carreras ordenadas por puntaje con nombre y puntuación
    print("Carreras recomendadas (ordenadas por puntaje):")
    for score, carrera_dict in career_scores:
        nombre_carrera = carrera_dict["nombre_carrera"]
        print(f"Carrera: {nombre_carrera}, Puntaje: {score:.2f}")  # Muestra solo el nombre y la puntuación


# Llamar a la función para ejecutar la prueba de VocationLab
if __name__ == "__main__":
    user_dictionary = {}
    user_dictionary = makeTest(user_dictionary)
    VocationLabTest(user_dictionary)
