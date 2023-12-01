# wolfram alpha API
import wolframalpha
import os
from pathlib import Path

from dotenv import load_dotenv
import openai

# Still have no idea why this works but hey I'm not complaining
path = Path("Environment-Variables/.env")
load_dotenv(dotenv_path=path)

# Setting the API Key
client = wolframalpha.Client(os.getenv("wlf_appid"))


# Quick and dirty way of solving equations
def solveEquation(equation):
    response = client.query(equation)
    return next(response.results).text
