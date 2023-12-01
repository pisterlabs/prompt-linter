# Script
import json
from matplotlib import pyplot as plt
import openai
import os
from scipy.stats import entropy, norm

import requests
from dotenv import load_dotenv
import chess

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


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


def getStockfish(fen):
    STOCKFISH_PORT = 8002
    SERVER = f"http://127.0.0.1:{STOCKFISH_PORT}"
    ROUTE = "/getAnalysis"
    PARAMS = "?fen=" + fen

    response = requests.get(SERVER + ROUTE + PARAMS)
    return response.json()


def getEval(fen):
    STOCKFISH_PORT = 8002
    SERVER = f"http://127.0.0.1:{STOCKFISH_PORT}"
    ROUTE = "/getEval"
    PARAMS = "?fen=" + fen

    response = requests.get(SERVER + ROUTE + PARAMS)
    return response.json()

# Function to calculate entropy


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


san = "1. e4 e5 2. Nf3 Nc6 3. c3 d5 4. d3 dxe4 5. dxe4 Qxd1+ 6. Kxd1 Be6 7. Ng5 Bg4+ 8. f3 Bd7 9. Bc4 Nh6 10. Kc2 Na5 11. Be2 f6 12. Nh3 Bxh3 13. gxh3 Nf7 14. Nd2 g6 15. h4 Bh6 16. h5 O-O-O 17. hxg6 hxg6 18. b4 Nc6 19. h4 Bg5 20. Nc4 Bxh4 21. Be3 Kb8 22. b5 Ne7 23. a4 Bg5 24. Bf2 Rxh1 25. Rxh1 Rh8 26. Re1 Rh2 27. Bg1 Rg2 28. Kd3 Nd8 29. a5 Ne6 30. b6 Nf4+ 31. Kc2 Rxe2+ 32. Rxe2 Nxe2 33. bxa7+ Ka8 0-1"

board = chess.Board()
states = []
fen = board.fen()
blunder = getBlunder(fen)
print(blunder)
states.append(
    {"blunder_first": blunder['first'], "blunder_second": blunder['second'], "fen": fen})
for move in san.split(" "):
    if move[0].isnumeric():
        continue
    board.push_san(move)
    fen = board.fen()
    blunder = getBlunder(fen)
    states.append(
        {"blunder_first": blunder['first'], "blunder_second": blunder['second'], "fen": fen})

# sort by blunder first, top 5 and bottom 5
states = sorted(states, key=lambda k: k['blunder_first'], reverse=True)
print("Top 5 blunder first:")
for state in states[:5]:
    print(state)

print("Bottom 5 blunder first:")
for state in states[-5:]:
    print(state)

print()
# sort by blunder second, top 5 and bottom 5
states = sorted(states, key=lambda k: k['blunder_second'], reverse=True)
print("Top 5 blunder second:")
for state in states[:5]:
    print(state)

print("Bottom 5 blunder second:")
for state in states[-5:]:
    print(state)