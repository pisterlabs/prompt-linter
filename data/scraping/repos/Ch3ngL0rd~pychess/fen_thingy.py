# Last thing before I sleep!
# Blunder Likelihood (High/Low) vs. Complexity (High/Low)

import json
from matplotlib import pyplot as plt
import openai
import os
from scipy.stats import entropy, norm

import requests
from dotenv import load_dotenv
import chess

def getMaia(fen):
    MAIA_PORT = 8001
    SERVER = f"http://127.0.0.1:{MAIA_PORT}"
    ROUTE = "/getMove"
    PARAMS = "?fen=" + fen

    request = requests.get(SERVER + ROUTE + PARAMS)
    response = request.json()
    # sorts the moves by probability
    response["maia-1100"]["moves"] = {k: v for k, v in sorted(
        response["maia-1100"]["moves"].items(), key=lambda item: item[1], reverse=True)}
    response["maia-1900"]["moves"] = {k: v for k, v in sorted(
        response["maia-1900"]["moves"].items(), key=lambda item: item[1], reverse=True)}
    return response

def calculate_entropy(distribution):
    # Get the probabilities
    probabilities = list(distribution.values())
    # Calculate and return the entropy
    return entropy(probabilities, base=2)

def getComplexity(maiaEval):
    MAIA_1100_STD = 0.8624
    MAIA_1900_STD = 0.8478
    MAIA_1100_MEDIAN = 3.4772
    MAIA_1900_MEDIAN = 2.9330
    # Calculate the entropy for each position and each Maia version
    entropy_value_1100 = calculate_entropy(maiaEval["maia-1100"]["moves"])
    entropy_value_1900 = calculate_entropy(maiaEval["maia-1900"]["moves"])

    # Calculates the complexity of the position
    complexity_1100 = norm.cdf(
        (entropy_value_1100 - MAIA_1100_MEDIAN) / MAIA_1100_STD)
    complexity_1900 = norm.cdf(
        (entropy_value_1900 - MAIA_1900_MEDIAN) / MAIA_1900_STD)

    return complexity_1100, complexity_1900


def getBlunder(fen):
    # Returns the blunder score
    BLUNDER_PORT = 8003
    SERVER = f"http://127.0.0.1:{BLUNDER_PORT}"
    ROUTE = "/getBlunder"
    PARAMS = "?fen=" + fen

    response = requests.get(SERVER + ROUTE + PARAMS)
    return response.json()

with open("data/random_fens.json") as f:
    data = json.load(f)

datas = []
# calculates the blunder and complexity for each fen
for index,fen in enumerate(data):
    print(f"{index}/{len(data)}")
    if index == 2000:
        break
    maiaEval = getMaia(fen)
    complexity_1100, complexity_1900 = getComplexity(maiaEval)
    blunder = getBlunder(fen)
    datas.append({
        "fen": fen,
        "complexity": complexity_1100 + complexity_1900 / 2,
        "blunder": blunder["blunder_chance"]
    })

# saves the data
with open("data/complexity.json", "w") as f:
    json.dump(datas, f, indent=4)