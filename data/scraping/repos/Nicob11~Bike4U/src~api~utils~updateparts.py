import requests
import json
from api.models import db, BikePart, Bike
import openai
import os
from time import sleep
from flask import jsonify
from bs4 import BeautifulSoup

GPT = os.getenv("OPENAI_API_KEY")
ORGANIZATION = os.getenv("ORGANIZATION")
BIKES = os.getenv("BIKES")
IMG_DEFAULT = os.getenv("IMG_DEFAULT")
#BIKES#
MTB = os.getenv("MTB")
ROAD = os.getenv("ROAD")
URBAN = os.getenv("URBAN")

#MTB#
FRAME_MTB = os.getenv("FRAME_MTB")
HANDLEBAR_MTB = os.getenv("HANDLEBAR_MTB")
PEDEDALS_CHAIN_MTB = os.getenv("PEDEDALS_CHAIN_MTB")
SADDLE_MTB = os.getenv("SADDLE_MTB")
FORKS_S = os.getenv("FORKS_S")
FORKS_M = os.getenv("FORKS_M")
FORKS_L = os.getenv("FORKS_L")
#ROAD#
FRAME_ROAD = os.getenv("FRAME_ROAD")
HANDLEBAR_ROAD = os.getenv("HANDLEBAR_ROAD")
PEDEDALS_CHAIN_ROAD = os.getenv("PEDEDALS_CHAIN_ROAD")
RIGID_FORKS = os.getenv("RIGID_FORKS")
SADDLE_ROAD = os.getenv("SADDLE_ROAD")
#URBAN#
FRAME_URBAN = os.getenv("FRAME_URBAN")
HANDLEBAR_URBAN = os.getenv("HANDLEBAR_URBAN")
PEDEDALS_CHAIN_URBAN = os.getenv("PEDEDALS_CHAIN_URBAN")
SADDLE_URBAN = os.getenv("SADDLE_URBAN")
RIGID_FORKS = os.getenv("RIGID_FORKS")
#WHEELS#
WHEEL_S = os.getenv("WHEEL_S")
WHEEL_M = os.getenv("WHEEL_M")
WHEEL_L = os.getenv("WHEEL_L")

parts_json = "src/api/utils/parts.json"
bikes_json = "src/api/utils/bikes.json"

def load_from_json(archivo_json):
    try:
        with open(archivo_json, "r") as infile:
            data = json.load(infile)
    except FileNotFoundError:
        data = []
    except json.decoder.JSONDecodeError:
        data = []
    return data

def save_to_json(data, archivo_json):
    try:
        existing_data = load_from_json(archivo_json)
    except FileNotFoundError:
            existing_data = []
    existing_data.extend(data)
    with open(archivo_json, "w") as outfile:
        json.dump(existing_data, outfile, indent=4, ensure_ascii=False)


def steal_parts(part, terrain, size):
    all_parts = []
    bike_parts_url = {
        "frame": {
            "mtb":{
                "s": FRAME_MTB,
                "m": FRAME_MTB,
                "l": FRAME_MTB
            },
            "road":{
                "s": FRAME_ROAD,
                "m": FRAME_ROAD,
                "l": FRAME_ROAD
            },
            "urban":{
                "s": FRAME_URBAN,
                "m": FRAME_URBAN,
                "l": FRAME_URBAN
            }
        },
        "wheels": {
            "mtb":{
                "s": WHEEL_S,
                "m": WHEEL_M,
                "l": WHEEL_L
            },
            "road":{
                "s": WHEEL_S,
                "m": WHEEL_M,
                "l": WHEEL_L
            },
            "urban":{
                "s": WHEEL_S,
                "m": WHEEL_M,
                "l": WHEEL_L
            }
            },
        "handlebar": {
            "mtb":{
                "s": HANDLEBAR_MTB,
                "m": HANDLEBAR_MTB,
                "l": HANDLEBAR_MTB
            },
            "road":{
                "s": HANDLEBAR_ROAD,
                "m": HANDLEBAR_ROAD,
                "l": HANDLEBAR_ROAD
            },
            "urban":{
                "s": HANDLEBAR_URBAN,
                "m": HANDLEBAR_URBAN,
                "l": HANDLEBAR_URBAN
            }
            },
        "pedals_chain": {
            "mtb":{
                "s": PEDEDALS_CHAIN_MTB,
                "m": PEDEDALS_CHAIN_MTB,
                "l": PEDEDALS_CHAIN_MTB
            },
            "road":{
                "s": PEDEDALS_CHAIN_ROAD,
                "m": PEDEDALS_CHAIN_ROAD,
                "l": PEDEDALS_CHAIN_ROAD
            },
            "urban":{
                "s": PEDEDALS_CHAIN_URBAN,
                "m": PEDEDALS_CHAIN_URBAN,
                "l": PEDEDALS_CHAIN_URBAN
            }
            },
        "saddle": {
            "mtb":{
                "s": SADDLE_MTB,
                "m": SADDLE_MTB,
                "l": SADDLE_MTB
            },
            "road":{
                "s": SADDLE_ROAD,
                "m": SADDLE_ROAD,
                "l": SADDLE_ROAD
            },
            "urban":{
                "s": SADDLE_URBAN,
                "m": SADDLE_URBAN,
                "l": SADDLE_URBAN
            }
            },
        "forks": {
            "mtb":{
                "s": FORKS_S,
                "m": FORKS_M,
                "l": FORKS_L
            },
            "road":{
                "s": RIGID_FORKS,
                "m": RIGID_FORKS,
                "l": RIGID_FORKS
            },
            "urban":{
                "s": RIGID_FORKS,
                "m": RIGID_FORKS,
                "l": RIGID_FORKS
            }
            },

        }
    if part not in bike_parts_url or terrain not in bike_parts_url[part] or size not in bike_parts_url[part][terrain]:
        return jsonify({"msg": "Frame type not found"}), 404
    if bike_parts_url[part][terrain][size] == None:
        url = BIKES
    else:
        url = BIKES + bike_parts_url[part][terrain][size]
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser').find('div', class_='items')
    parts = soup.find_all('a', 'item site-hover site-product-list-item-nojs')
    all_url = []
    for href in parts:
        url = "https://www.bike-components.de" + href['href']
        all_url.append(url)
    for url in all_url:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser').find('article', class_='container module-product-detail js-site-init-functions site-module-margin-bottom')
        parts = soup.find('div', class_='row')
        imagen = parts.find("div", class_="image").find("img").get("src")
        if imagen == None:
            img = IMG_DEFAULT
        else:
            img = "https://www.bike-components.de" + imagen
        title = parts.find("li", class_="flex items-center grow md:w-full md:pt-4").find("h1").text.strip()
        url = url
        terrain = terrain.lower()
        size = size.lower()
        description = soup.find("div", class_="description").find("div", class_="site-text").find("h2")
        if description == None:
            description = "No description"
        else:
            description = description.text.strip()
        new_part = {
            "part":part,
            "terrain":terrain,
            "size":size,
            "title":title,
            "image":img,
            "description":description,
            "link":url
        }
        all_parts.append(new_part)
    save_to_json(all_parts, parts_json)
    return jsonify({"msg": "Frames added"}), 200


def steal_bikes(terrain):
    all_bikes = [] 
    bikes = {
        "mtb": MTB,
        "road": ROAD,
        "urban": URBAN
    }
    if terrain not in bikes:
        return jsonify({"msg": "Terrain not found"}), 404
    url = bikes[terrain]
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser').find_all('a', class_='item site-hover site-product-list-item-nojs')
    bikes = []
    for bike in soup:
        bike_url = "https://www.bike-components.de" + bike['href']
        bikes.append(bike_url)
    for bike in bikes:
        response = requests.get(bike)
        soup = BeautifulSoup(response.text, 'html.parser').find('div', id='wrapper')
        bikes = soup.find('div', class_='row')
        imagen = bikes.find("div", class_="image").find("img").get("src")
        if imagen == None:
            img = IMG_DEFAULT
        else:
            img = "https://www.bike-components.de" + imagen
        title = bikes.find("li", class_="flex items-center grow md:w-full md:pt-4").find("h1").text.strip()
        url = bike
        terrain = terrain.lower()
        description = soup.find("div", class_="description").find("div", class_="site-text").find("h2")
        if description == None:
            description = "No description"
        else:
            description = description.text.strip()
        new_bike = {
            "title":title,
            "image":img,
            "link":url,
            "terrain":terrain,
            "description":description
        }
        all_bikes.append(new_bike)
    save_to_json(all_bikes, bikes_json)
    return jsonify({"msg": "Bikes taked"}), 200










# def use_gpt3(message):
#     response = openai.Completion.create(engine="text-davinci-003",
#                                         prompt=message,
#                                         temperature=0.7,
#                                         max_tokens=600,
#                                         top_p=1,
#                                         frequency_penalty=0,
#                                         presence_penalty=0
#                                         )
#     return response['choices'][0]['text']


# def get_part():
#     url = "https://bpartcomponents.com/wp-json/wp/v2/product?_fields=id,title,link,_embedded,_links&_embed"   
#     response = requests.get(url)
#     response.encoding = 'utf-8-sig'
#     data = json.loads(response.text)
#     list_data = [{'img': element['_embedded']['wp:featuredmedia'][0]['link']
#                   ,'title':element['title']['rendered'], 'link': element['link']} for element in data]
#     # diccionarios = []
#     # ejemplo = {'tamaño': '', 'link': '', 'tittle': ''}
#     # for element in list_data:
#     #     message = f"""
#     #     Rellena el siguiente objeto JSON de ejemplo con la informacion que te paso acontinuacion.
#     #     Solo quiero que devuelvas un objeto como el del ejemplo rellenada con la informacion mas abajo sin saltos de linea:
#     #     Ejemplo:
#     #     {ejemplo}
#     #     Informacion:{element}
#     #     """
#     #     part = use_gpt3(message)
#     #     print(part)
#     #     diccionarios.append(part)
#     #     sleep(1)
#     return list_data