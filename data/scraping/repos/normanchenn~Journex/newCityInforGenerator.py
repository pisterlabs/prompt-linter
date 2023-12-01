import cohere
import openai
import time
import json
import sys
from cohere.responses.classify import Example
co = cohere.Client('8GEb0w8gJW7TnKs6FLR45yWG0KSLIJNpeuIikuVF')
openai.api_key = "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"

f = open("ListCountries.txt", "r")
for country in f:
    tempPrompt = "List me the top 10 tourist attractions in " + country + " without description"
    print(tempPrompt)
    tempResponse = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user",
        "content": tempPrompt,}
        ]
    )
    content = tempResponse.choices[0].message.content
    items = content.split("\n")

    g = open("ListCountryAttractions.txt", "a")
    g.write(country)
    for outItem in items:
        g.write(outItem + "\n")
    g.write("\n")
    g.close()
    time.sleep(20)