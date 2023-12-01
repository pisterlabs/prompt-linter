_author1__ = 'Michael Ofengenden'
__copyright1__ = 'Copyright 2022, Michael Ofengenden'
__email1__ = 'michaelofengend@gmail.com'
import csv
import pandas as pd
#import openpyxl
from numpy import *
from collections import *
import os
import openai
from openai.embeddings_utils import get_embedding
import requests
import json

from flask import Flask, redirect, render_template, request, url_for

if __name__ != '__main__':
    import PhysicsBasics as pb



def AnalyzePhase(AtPct=None, WtPct=None, OxWtPct=None, OByStoich=None, APIkey=None):
    AtPct = AtPct/sum(AtPct)*100
    OutStr = '--- GPT-4 Analysis ---\n\n'
    #KnownElements = ['O', 'Na', 'Mg', 'Al', '
    # Si', 'P', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni']    
    #Pop up personal GPT-4 API login 
    API_KEY = APIkey
    API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'



    def generate_chat_completion(message, model="gpt-4", temperature=0.8, max_tokens=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        data = {
            "model": model,
            "messages": message,
            "temperature": temperature,
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error {response.status_code}: {response.text}"
    

    if APIkey:
        OutStr += "INPUT DATA:\n"
        GPTInput = "Given the following atomic weight percentages, what is the most likely mineral?\n\n"
        for Element in pb.ElementalSymbols[1:]:
            atomic= eval('AtPct[pb.%s-1]'%Element)
            if atomic != 0:
                # OutStr += f"{Element}: {atomic} \n"
                GPTInput += f"{Element}: {atomic} \n"
        message = [
            {"role": "system", "content": "You are an expert in mineralogy. You will give an elaborate response of mineral analysis "},
            {"role": "user", "content": GPTInput},
        ]

        reponse_text = generate_chat_completion(message)
        OutStr += reponse_text
    
    else:
        OutStr += "There is no API Key input"
    return OutStr, None




if __name__ == '__main__':
    import imp
    pb = imp.load_source('PhysicsBasics', '../PhysicsBasics.py')
    AnalyzePhase()