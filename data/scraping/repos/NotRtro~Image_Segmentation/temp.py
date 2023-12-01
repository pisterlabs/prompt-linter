from openai import OpenAI
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import webcolors
import numpy as np
import cv2
import requests

API_KEY = 'AIzaSyCJv3vyFYKSjxczFtTnGb7vuZvYj6OSQbY'
CX = '31b7410374d974eb9'  # Puedes configurar esto al crear tu motor de búsqueda personalizado

def buscar_imagenes(query):
    
    url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX}&searchType=image'
    response = requests.get(url)
    data = response.json()
    return data['items']



def dezglozardic(dic):
    final = []
    for key, value in dic.items():
        final.append(f'{key} con un total de {value} apariciones')
    return ', '.join(final)


def get_main_colors(imagenes, n_colors):
    result = []
    for i in imagenes:
        #image = cv2.imdecode(i, cv2.IMREAD_COLOR)
        try:
            image = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            image = image.reshape(-1, 3)

            kmeans = KMeans(n_clusters=n_colors, n_init=10)
            labels = kmeans.fit_predict(image)  

            counts = np.bincount(labels)
            colors = kmeans.cluster_centers_

            result.extend([rgb_to_name(colors[i].astype(int)), counts[0] / len(labels) * 100] for i in range(n_colors))
        except:
            pass
    result = sorted(result, key=lambda x: x[1], reverse=True)
    print(result)
    final = ''
    for i in result:
        final += i[0] + ', '
    return final, result



def rgb_to_name(rgb_color):
    try:
        color_name = webcolors.rgb_to_name(rgb_color)
    except ValueError:
        color_name = webcolors.rgb_to_hex(rgb_color)
    return color_name

client = OpenAI(api_key='')



def getKeywors(text):
    nltk.download('punkt')
    nltk.download('stopwords')

    stop_words = set(stopwords.words('spanish')) 
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ', '.join(filtered_sentence)


def generateCamp(caracteristicas, title, rubro, recurrencias, keyword):

    data =  f"""quiero que me generas ideas sobre campañas creativas y atractivas en el rubro {rubro} y 
    teniendo en cuenta que haciendo un análisis de los alrededores de mi negocio tanto en las personas 
    que pasan por mi negocio, y como son los alrededores de este, tenemos en cuenta que la mayoria de clientes o cosas relevantes al rededor de la tienda tienen 
    caracteristicas principales captados como {recurrencias}, y nuestro negocio tiene caracteristicas como {keyword}, 
    teniendo en cuenta que estas detonan colores principales dentro de los clientes como {caracteristicas} y que la temática es {title}. """

    print(data)# Fixed the formatting of the string
    
    response = client.chat.completions.create(model = "gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": """Eres un asistente de marketing que ayuda a 
                                        las personas a crear campañas publicitarias para 
                                        sus negocios con ideas creativas y factibles. quiero que seas detallado y preciso en cada idea que proporcionas para que me des una mejor idea de como llevarlas a cabo, 
                                        Retornas un JSON con los keys DESCRIPCION, NOMBRES como una lista, COLABORACIONES como una lista, IDEAS como una lista, los values deben ser solo texto limpio 
                                        no me des mas texto que el json, sin puntuaciones ni el ```json que pones al inicio y al final
                                        """},
        {"role": "user", "content": '1. Que ideas para campañas me recomiendas en base a las siguientes caracteristicas que quiero tomar en cuen ' + data + ' 2. Que nombres para una campaña puedo ponerle con la tematica' + title},
        #{"role": "user", "content": 'Que nombres para una campaña puedo ponerle con la tematica' + title},
    ])

    # Extrae el contenido de la respuesta
    content = response.choices[0].message.content
    # Elimina los caracteres de escape y las comillas adicionales
    content = content.replace("\\", "").strip('""')
    # Convierte la cadena de texto en formato JSON a un objeto Python
    campaign = json.loads(content)

    return campaign