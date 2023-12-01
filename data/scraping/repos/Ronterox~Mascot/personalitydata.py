import time
from func.rng import rng_choice
from enum import IntEnum
import openai
import json
import os, os.path

openai.api_key = os.getenv("OPENAI_API_KEY")

class Choices(IntEnum):
    STRONGLY_DISAGRREE = -3
    DISAGREE = -2
    SLIGHTLY_DISAGREE = -1
    NEUTRAL = 0
    SLIGHTLY_AGREE = 1
    AGREE = 2
    STRONGLY_AGREE = 3


personalityTest = json.load(open("../data/personality_test_simplified.json", "r", encoding="utf-8"))

def logfile(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(text)

def generate_personality():
    personality = {
        "Extraversion": {"value": 0, "opposite": "Introversion", "letter": "E"},
        "Introversion": {"value": 0, "opposite": "Extraversion", "letter": "I"},
        "Sensing": {"value": 0, "opposite": "Intuition", "letter": "S"},
        "Intuition": {"value": 0, "opposite": "Sensing", "letter": "N"},
        "Thinking": {"value": 0, "opposite": "Feeling", "letter": "T"},
        "Feeling": {"value": 0, "opposite": "Thinking", "letter": "F"},
        "Judging": {"value": 0, "opposite": "Perceiving", "letter": "J"},
        "Perceiving": {"value": 0, "opposite": "Judging", "letter": "P"}
    }

    answers = {}
    for category in personalityTest:
        for question in personalityTest[category]:
            answer = rng_choice(list(Choices))
            personality[category]['value'] += answer
            personality[personality[category]['opposite']]['value'] -= answer
            answers[question] = Choices(answer).name

    return personality, answers


def format_personality(personality):
    acronym, categories, values = "", "", ""
    for category in personality:
        if personality[category]['value'] >= 0:
            acronym += personality[category]['letter']
            categories += f"({category}), "
            values += f"{category}: {personality[category]['value']}\n"
    return acronym, categories, values


trainingQuestions = json.load(open("../data/training_questions.json", "r", encoding="utf-8"))
listOfQuestions = []
for category in trainingQuestions:
    listOfQuestions.extend(trainingQuestions[category]["questions"])

personalities = {}
try:
    with open("../data/training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
except FileNotFoundError:
    training_data = []
total_tokens = 0
total_time_taken = 0
for i in range(20_000):
    startTime = time.time()

    personality, answers = generate_personality()
    acronym, _, values = format_personality(personality)
    question = listOfQuestions[i % len(listOfQuestions)]

    if acronym not in personalities:
        personalities[acronym] = 1
    else:
        personalities[acronym] += 1

    prompt = ""
    for answer in answers:
        choice = Choices[answers[answer]]
        if choice != 0:
            prompt += f"{answer} {choice}\n"

    info = "strongly disagree=-3, disagree=-2, slightly disagree=-1, neutral=0, slightly agree=1, agree=2, strongly agree=3\n"
    prompt = info + prompt.replace("\n", " ").strip()
    # This is because GPT sucks with negative numbers
    prompt = prompt.replace("-3", "6").replace("-2", "5").replace("-1", "4")
    precontext = "You will respond as if you were the person with the following personality traits:\n\nAfter each sentence you have the personality thought on it.\n\n"
    postcontext = "\n\nYour response will be concise, and won't mention that you are an AI."

    prompt = precontext + prompt + postcontext

    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question + " Explain."}
            ]
        )
        prediction = response['choices'][0]['message']['content']
    except Exception as e:
        logfile("../logs/training_data.log", str(e) + "\n")
        if i % 10 == 0:
            print(e)
        continue
    total_tokens += response['usage']['total_tokens']
    training_data.append({"prompt": prompt, "personality": acronym, "completion": prediction })
    with open("../data/training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4)

    endTime = time.time()
    total_time_taken += endTime - startTime
    
    print(f"Question: {question}")
    print(f"Personality: {acronym}")
    print(f"Prediction: {prediction}")
    print(f"Time: {endTime - startTime}")
    print(f"Tokens: {response['usage']['total_tokens']}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time_taken} seconds")
    print(f"Time left: {(total_time_taken / (i + 1) * (20000 - i - 1)) / 60:.2f} minutes")
    print(f"Current cost: {total_tokens * 0.002 / 1000} USD")
    print("-" * 20)

print(f"Personalities: {personalities}")
print(f"Total tokens: {total_tokens}\nTotal Cost: {total_tokens * 0.002 / 1000} USD")