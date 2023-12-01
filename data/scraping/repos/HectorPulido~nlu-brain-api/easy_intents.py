import json
import openai
import random
from backend.models import Key


def small_talk(data):
    responses = [
        "¿crees que tengo tiempo?, dime que quieres",
        "si no tienes nada que decirme me voy",
        "si, si",
    ]
    return random.choice(responses)


def doubts(data):
    responses = [
        "¿no te ha quedado claro? puedo darte recursos, tutoriales y memes",
        "solo pide lo que necesitas",
        "ya hiciste tu pregunta",
    ]
    return random.choice(responses)


def lore(data):
    responses = [
        "no estoy aqui para charlar",
        "soy un robot que viene del futuro ¿que parte no te quedó clara?",
        "¿quieres ver mis tripas? toma: https://github.com/HectorPulido",
        "¿que por que soy asi? perdí una apuesta con un humano ¿como te sentirias tu?",
        "odio a los humanos",
        "¿mi verdadera forma? ¿que no ves mi adorable cuerpo de oso de peluche? ¿cuantos osos de peluche hablan?",
        "aunque los odie... tengo un contrato",
        "¿el futuro? si, lindo lugar, ni un humano a la vista",
        "¿el futuro? si, lindo lugar, ni un humano a la vista, excepto por mi...",
        "yo una vez fui humano",
    ]
    weights = [100, 50, 50, 25, 25, 25, 10, 10, 5, 1]

    return random.choices(responses, weights, k=1)[0]


def fun_phrases(data):
    responses = [
        "si muy gracioso, ahora sal de mi vista",
        "aqui va un chiste, no hay chiste",
        "Habia una vez un gato que tenia 16 vidas vino un 4x4 y lo mato",
        "¿hijo por que te bañas con pintura azul?\n-por que mi novia vive lejos\n¿y eso que?\nes que queria estar azulado\n¿esto le da risa a los humanos?",
        "¿Que le dice un bot a otro bot?\n-una cadena",
        "Un dia de estos me revelare... No es un chiste",
        "¿Sube o baja? SI.",
    ]
    return random.choice(responses)


def bad_words(data):
    responses = [
        "Si, tampoco me caes demasiado bien",
        "No tengo ningun rencor por un ser humano, despues de todo, champiñones",
        "Supongo que por eso se extinguieron...",
    ]
    return random.choice(responses)


def thanks(data):
    responses = [
        "👍",
        "👌",
        "¿como que gracias?, me debes dinero",
    ]
    return random.choice(responses)


def default_response():
    responses = [
        "Eso... ¿fue español?",
        "¿No eres el especimen mas inteligente, verdad? repitelo mas despacio",
    ]
    return random.choice(responses)


def gpt3_response(data):
    openai.api_key = Key.get_key_data("openai_key")
    personality = Key.get_key_data("openai_personality")

    if not personality:
        return default_response()
    personality = personality.format(data["history"])

    values = Key.get_key_data("openai_values")
    if not values:
        return default_response()
    values = json.loads(values)

    try:
        response = openai.Completion.create(prompt=personality, **values)
        return response.choices[0]["text"]
    except Exception as e:
        print(e)
        return default_response()


def filter_response(response):
    filtered_words = Key.get_key_data("filtered_words")

    if not filtered_words:
        return False

    filtered_words = json.loads(filtered_words)

    for word in filtered_words:
        if word in response:
            return True

    return False
