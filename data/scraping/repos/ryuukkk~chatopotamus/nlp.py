import pickle
import random
import threading
from openai import OpenAI
from src.training_and_prediction import predict, models
import audio_mgmt

print('NOW LOADING NER...')
with open('resources/bert/saved/ner_tokenizer.pkl', 'rb') as tn:
    bert_ner_tokenizer = pickle.load(tn)
with open('resources/bert/data/label_map.pkl', 'rb') as lm:
    label_map = pickle.load(lm)
bert_ner_model = models.trained_entity_classifier()
bert_ner_model.load_weights('resources/bert/saved/ner_trained_weights.h5')
print('LOADED NER')

print('\b' * 18)
print('LOADING INTENT...')
with open('resources/bert/saved/intent_tokenizer.pkl', 'rb') as ti:
    bert_intent_tokenizer = pickle.load(ti)
bert_intent_model = models.trained_intent_classifier()
bert_intent_model.load_weights('resources/bert/saved/ir_trained_weights.h5')
print('LOADED INTENT')

print('\b' * 18)
print('NOW LAODING GPT...')
with open('resources/API_KEY.txt', 'r') as f:
    OPENAI_API_KEY = f.readline()
OPENAI_JOB = "ftjob-NOjJ5NxYigdba5FCHz8GXwQo"
GPT3_MODEL = "ft:gpt-3.5-turbo-0613:personal::8PhccnUL"
client = OpenAI(api_key=OPENAI_API_KEY)
# completion = client.fine_tuning.jobs.retrieve(OPENAI_JOB)

menu = {
    "prices": {"coffee": 1.50, "cappuccino": 2.50, "iced coffee": 2, "iced capp": 2.25, "latte": 2, "tea": 1.50,
               "hot chocolate": 2.25, "french vanilla": 2.25, "white chocolate": 2.25,
               "mocha": 2.25, "espresso": 1, "americano": 2.25, "extra shot": 0.25, "soy milk": 0.3,
               "whipped topping": 1, "dark roast": 0.20, "Turkey Bacon Club": 3, "BLT": 2.90,
               "grilled cheese": 4, "chicken wrap": 3.50, "soup": 2.80, "donut": 1.5, "double double": 1.50,
               "triple triple": 1.50, "muffin": 2.40, "bagel": 3, "timbits": 3, "panini": 2.40, "croissant": 3},
    "price multiplier": {"small": 1, "medium": 1.2, "large": 1.4, "extra large": 1.6}
}


def get_all_info(request):
    intent = predict.predict_intent(request, model=bert_intent_model, tokenizer=bert_intent_tokenizer)
    entities = predict.predict_entities(request, model=bert_ner_model, tokenizer=bert_ner_tokenizer,
                                        label_map=label_map, max_seq_length=26)
    response, message = predict.chat_with_assistant(request, client=client, model=GPT3_MODEL, fresh=False)

    return intent, entities, (response, message)


def regular_customer(opening):
    messages = []
    intents = []
    entity_tags = []
    response = None
    total_price = 0

    audio_mgmt.speak(opening)
    while True:
        request = audio_mgmt.speech_to_text()

        if not request or len(request) < 4:
            audio_mgmt.speak('Visit again, Bye!')
            break
        response, messages = predict.chat_with_assistant(request, messages=messages, client=client, model=GPT3_MODEL)
        intent = predict.predict_intent(request, model=bert_intent_model, tokenizer=bert_intent_tokenizer)
        entities = predict.predict_entities(request, model=bert_ner_model, tokenizer=bert_ner_tokenizer,
                                            label_map=label_map,
                                            max_seq_length=26)

        order_price = get_price(entities, 0)
        total_price += order_price
        response = response.replace("<price>", str(total_price))

        audio_mgmt.speak(response)
        intents.append(intent)
        print(f'{intent} : '.upper())

        entity_tags.append(entities)
        print_formatted_entities(entities)

        if 'order' in map(str.lower, intents):
            print('Total: $' + str(total_price))

    return intents, entity_tags, (response, messages)


def new_customer(opening, face_encoding):
    intents, entity_tags, (response, messages) = regular_customer(opening)
    r = random.choice(['amm..', ''])
    audio_mgmt.speak(str(r) + 'One last thing before we see you again, would you like to tell me your name if you want '
                              'me to remember you when you visit next time?')
    response_2 = audio_mgmt.speech_to_text().lower()
    audio_mgmt.speak('alright, visit again, bye')
    return intents, entity_tags, (response, messages), ''


def get_price(entities, current_price):
    global menu  # Assuming the menu dictionary is globally available

    total_price = current_price
    last_beverage_price = 0
    size_multiplier = 1

    for entity in entities:
        entity_value, entity_type = entity

        # Handling beverage items
        if 'beverage' in entity_type:
            # Apply previous multiplier to the last beverage and reset it
            total_price += last_beverage_price * size_multiplier
            last_beverage_price = menu["prices"].get(entity_value, 0)
            size_multiplier = 1  # Reset multiplier for new beverage

        # Check if the entity is a beverage size for multiplier
        elif entity_type == 'beverage_size':
            size_multiplier = menu["price multiplier"].get(entity_value, 1)

        # Handling non-beverage items
        else:
            item_price = menu["prices"].get(entity_value, 0)
            total_price += item_price

    # Apply multiplier to the last beverage in the list
    total_price += last_beverage_price * size_multiplier

    return total_price


def print_formatted_entities(entities):
    if not entities:
        return

    beverage_line = ""
    food_line = ""

    for entity_value, entity_type in entities:
        if 'beverage' in entity_type:
            beverage_line += f" {entity_value} " if beverage_line else entity_value
        elif 'food' in entity_type:
            food_line += f" {entity_value} " if food_line else entity_value

    # Print formatted lines
    if beverage_line:
        print("Beverage: " + beverage_line.title())
    if food_line:
        print("Food: " + food_line.title())


