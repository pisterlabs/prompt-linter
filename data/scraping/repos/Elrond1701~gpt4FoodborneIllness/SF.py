import openai
from EMD import EMD
from util import ASST, MODEL, USER

question1 = "I'm an excellent linguist. The task is to check the whether the giving entity in the given sentences is "
question2 = "the food that caused the foodborne illness. "
question3 = "the place that caused the foodborne illness. "
question4 = "Please only give yes or no."


def isWhat(sentence, entity):
    messages = []
    messages.append({
        "role": USER,
        "content": question1 + question2 + question4
    })
    response = openai.ChatCompletion.create(MODEL, messages)

    answer = response["choices"][0]["text"]
    messages.append({
        "role": ASST,
        "content": answer
    })
    messages.append({
        "role": USER,
        "content": "Entity: " + entity + "\r\n" + "Sentences: " + sentence
    })
    response = openai.ChatCompletion.create(MODEL, messages)
    answer = response["choices"][0]["text"]
    if answer.find("yes") != -1:
        return True
    elif answer.find("no") != -1:
        return False
    else:
        raise Exception()


def isWhere(sentence, entity):
    messages = []
    messages.append({
        "role": USER,
        "content": question1 + question3 + question4
    })
    response = openai.ChatCompletion.create(MODEL, messages)

    answer = response["choices"][0]["text"]
    messages.append({
        "role": ASST,
        "content": answer
    })
    messages.append({
        "role": USER,
        "content": "Entity: " + entity + "\r\n" + "Sentences: " + sentence
    })
    response = openai.ChatCompletion.create(MODEL, messages)
    answer = response["choices"][0]["text"]
    if answer.find("yes") != -1:
        return True
    elif answer.find("no") != -1:
        return False
    else:
        raise Exception()


def SF(sentence: str, examples=False):
    entities = EMD(sentence, examples)
    symptom = entities[1][answer]

    what = entities[2][answer]
    what_ = []
    for entity in what:
        if isWhat(sentence, entity):
            what_.append(entity)
    what = what_

    where = entities[0][answer]
    where_ = []
    for entity in where:
        if isWhere(sentence, entity):
            where_.append(entity)
    where = where_

    answer = {
        "symptom": symptom,
        "what": what,
        "where": where
    }
    return answer