# Day 97 on Replit: "Challenge - WebScraping + OpenAI"

import requests
from bs4 import BeautifulSoup
import openai
import config

url = input("Paste wiki URL > ")

response = requests.get(url)
html = response.text
soup = BeautifulSoup(html, "html.parser")

main_text = soup.find_all("div", class_="mw-parser-output")

words = ""
for section in main_text:
    text = section.find_all("p")

    for p in text:
        words += p.text

references = soup.find_all("ol", class_="references")
references_link = references[0].find_all("a", class_="external text")

openai.organization = config.orgID_key
openai.api_key = config.openAI_key
openai.Model.list()

prompt = f"Make me a summary of the next text in no more than 3 paragraphs: '{text}'"

response_ai = openai.Completion.create(
    model="text-davinci-003", prompt=prompt, max_tokens=100, temperature=0)

summarize = response_ai["choices"][0]["text"]
print()
print(summarize.strip())
print()
print("References")
print()
print(references[0].text.replace("^ ", ""))
