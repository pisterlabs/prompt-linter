import os
import requests
import pandas as pd
import json
import time
import urllib.request
from montydb import MontyClient
import asyncio
import openai

from PyPDF2 import PdfReader

def is_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            pdf = PdfReader(file)
            _ = len(pdf.pages)
            print(_)
        return True
    except Exception as e:
        print(e)
        return False

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


def remove_surrogates(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')


import re

def extract_json_values_using_regex(input_string):
    json_pattern = r'<json>(.*?)</json>'
    matches = re.findall(json_pattern, input_string, re.DOTALL)
    if matches:
        json_str = matches[-1]  # Get the last match in case of multiple occurrences
        json_data = json.loads(json_str)
        return json_data
    else:
        return None


client = MontyClient("Verra.db")
db = client["Verra"]
project_reviews = db["ProjectReviews"]

anthropic = Anthropic(api_key='XXXX')

openai.api_key = 'XXX'

data = pd.read_csv("Verra_Projects.csv")

status_filter = ["Under Validation", "Registration requested", "Under development", 
                 "Registration and verification approval requested"]
filtered_data = data[data.Status.isin(status_filter)]


def print_first_10_lines(input_string):
    lines = input_string.split('\n')[:3]
    for line in lines:
        print(line)


def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            pdf = PdfReader(file)
            num_pages = len(pdf.pages)
            text = " ".join(page.extract_text() for page in pdf.pages)
            return text
    except Exception as e:
        print(f"Error: {e}")
        return None

for _, row in filtered_data.iterrows():
    project_id = row['ID']
    pdd_file = f"PDD/{project_id}.pdf"
    
    print("Preparing to fetch Project " + str(project_id))
    if os.path.isfile(pdd_file):
        print(f"File {pdd_file} already exists, skipping...")
        continue

    time.sleep(3)

    response = requests.get(f"https://registry.verra.org/uiapi/resource/resourceSummary/{project_id}")
    if response.status_code != 200:
        print(f"Warning: Could not fetch project details for project {project_id}")
        continue

    openaiResponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": 'You are an expert analyst in the voluntary carbon markets, with 30+ experience in Verra. You will be given a JSON. Your task is to extract the URL for the most updated version of the Project Design Document (PDD), by analysing the "documentType", "documentName" & "uploadDate"\nEXTREMELY IMPORTANT: YOU WILL ONLY OUTPUT URL. NO COMMENTS. \nSTART WITH: "https://"'}, 
            {"role": "user", "content": str(response.json())}
        ],
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    #print("Response from OpenAI")
    #print(openaiResponse)
    pdd_url = openaiResponse['choices'][0]['message']['content']

    if pdd_url.startswith("https://"):

        urllib.request.urlretrieve(pdd_url, pdd_file)        
        if is_pdf(pdd_file) is False:
            print(f"Warning: Downloaded file PDD/{project_id}.pdf is not a PDF, skipping")
            continue

        pdd_content = remove_surrogates(extract_text_from_pdf(pdd_file))

        print_first_10_lines(pdd_content)

        stream = anthropic.completions.create(
            prompt=f"{HUMAN_PROMPT} You are an expert auditor in the voluntary carbon markets, with over 30+ years of experience in Verra Methodologies. You will be provided with a Project Design Document (PDD).\n\n<document>{pdd_content}<document>\n\nYour task is to critically rate each section of the PDD and score it out of 10. If there are missing items, you should severely downgrade the project. You will be EXTREMELY STRICT with regards to Additionality & Permanance in particular, especially on regulatory surplus.\n\nWhen you reply, please provide your response in JSON format that satisfies the following structure:\n\n```type ProjectReview = {{\n    projectDetails: Detail,\n    safeguards: Detail,\n    applicabilityOfMethodology: Detail,\n    projectBoundary: Detail,\n    baseline: Detail,\n    additionality: Detail,\n    emissionReductions: Detail,\n    monitoringPlan: Detail,\n    implementationStatus?: Detail,\n    estimatedEmissionReductions?: Detail,\n    monitoring?: Detail,\n    quantificationOfEmissionReductions?: Detail,\n    overallScore: Detail\n}}\n\ntype Detail = {{\n    score: number | string,\n    comments: string\n}}\n\n```Please put your JSON response inside the <json></json> XML tags.\n{AI_PROMPT}",
            max_tokens_to_sample=2048,
            model="claude-2",
            stream=True
        )
        output_str = ""
        for completion in stream:
            print(completion.completion, end="")
            output_str += completion.completion

        try:
            json_data = extract_json_values_using_regex(output_str)
            # If parsing succeeds, the string is valid JSON.
            print(json_data)
            project_reviews.insert_one({"project_id": project_id, "project_review": json_data})
            print("Successfully inserted JSON data.")
        except json.JSONDecodeError as e:
            # If parsing fails, the string is not valid JSON.
            print(f"Error: The output_str is not valid JSON. {e}")
            # Optionally, you can log this error or handle it as needed
        
    else:
        print(f"Warning: Could not extract URL for project {project_id}")

client.close()
