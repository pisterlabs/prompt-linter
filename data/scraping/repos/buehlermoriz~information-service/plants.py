#keys
import json
import uuid
import requests
from modules import openai
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from modules import firebase


def plant_lookup(CLIENT, BUCKET, name: str, plant_id: str = None):
    #search for Plant in Firestore
    db = CLIENT
    plant_collection = db.collection('plants')

    if name:
        docs = plant_collection.get()
        # Perform fuzzy matching to find the closest match
        best_match = process.extractOne(name, [doc.get('common_name') for doc in docs])
        if best_match[1] > 85:
            plant_name = best_match[0]
            plant_doc = plant_collection.where('common_name', '==', plant_name).get()[0]
            plant = plant_doc.to_dict()
            return plant
        else:
            plant = generate_new_plant(name, CLIENT, BUCKET)
            return plant
    elif plant_id:
        plant_doc = plant_collection.where('id', '==', plant_id).get()[0]
        plant = plant_doc.to_dict()
        return plant
    else:
        return "ERROR - info_service - plants.py - plant_lookup: Es wurde weder ein Name noch eine ID angegeben. Bitte überprüfe deine Anfrage."

    
def plant_list_lookup(names: list, CLIENT):
    #search for Plant in Firestore
    db = CLIENT
    
    # perform a query for all plants matching the common names
    docs = db.collection('plants').where('id', 'in', names).get()

    #if plant is in the database
    if len(docs) > 0:
        plants = [doc.to_dict() for doc in docs]
        return plants
    #if plant is nowhere in the database
    else:
        return "ERROR - info_service - plants.py - plant_list_lookup: Die angefragten Pflanzen ("+str(names)+") einer der angefragten Beete existieren nicht! Wahrschienlich wurden diese manuell gelöscht. Bitte werfe einen Blick in den Firestore, oder melde dich bei mir (Moriz)", 400
    
def all_plants(CLIENT):
    #search for Plant in Firestore
    db = CLIENT
    
    # perform a query for all plants matching the common names
    docs = db.collection('plants').get()

    #if plant is in the database
    if len(docs) > 0:
        plants = [doc.to_dict() for doc in docs]
        return plants
    #if plant is nowhere in the database
    else:
        return "plants not found", 400
    

def generate_new_plant(name: str, CLIENT, BUCKET, id: str = None):
    #check if request is from reload plant. If it is from relaod plant, the provided id is used
    if id:
        plant_id = id
    else:
        plant_id = uuid.uuid4().hex
    #generate and upload new plant data
    plant = {
        "id": plant_id,
        "common_name": name,
    }
    prompt = 'Liefere mir ein Array mit Daten über: \n ' + name + '\n mit den Informationen: \n {"scientific_name": latein \n "description": maximal drei Sätze \n "harvest": Ein Wort aus Frühling, Sommer, Herbst, Winter \n "sun": ganzzahliger Wert zwischen 0 und 5 \n "water": ganzzahliger Wert zwischen 0 und 5 \n "ph": ganzzahliger Wert zwischen 0 und 14 \n "companion_plants": Aufzählung von Pflanzennamen getrennt mit einem Komma \n "toxic_level": wert zwischen 0 und 4 (0=nicht giftig, 1= wenig giftig, 2= gifitig, 3=stark giftig, 4=sehr stark gifitg) \n "taste": Wähle zwischen den Geschmacksrichtungen Umami, Süß, Sauer, Salzig, Bitter} \n Beachte die Formatierungsvorgaben nach dem jeweiligen Doppelpunkt. \n Gibt es mehrere Pflanzen mit dieser Bezeichnung wähle die am weitesten verbreitete aus. \n Liefere mir nur das Array und keine weiteren Informationen oder Text zurück.'
    ai_response = openai.request_open_ai(prompt)
    # parse the API response as a dictionary
    api_dict = json.loads(ai_response.replace('\n', '').replace('[', '').replace(']', ''))
    companion_plants = api_dict.get("companion_plants").split(',')

    # update the plant dictionary with the API data
    plant.update({
        "scientific_name": api_dict.get("scientific_name"),
        "description": api_dict.get("description"),
        "harvest": api_dict.get("harvest"),
        "sun": api_dict.get("sun"),
        "water": api_dict.get("water"),
        "ph": api_dict.get("ph"),
        "toxic_level": api_dict.get("toxic_level"),
        "taste": api_dict.get("taste"),
        "companion_plants": companion_plants
    })

    #get plant image
    image_url, plant_img = openai.request_open_ai_image(name)
    plant["firebase_path"], plant["img"] = firebase.upload_image(image_url, plant_img, "plantimages/",128, BUCKET)
    #push to firebase
    firebase.upload_plant(plant["id"], plant, "plants", CLIENT)
    return plant