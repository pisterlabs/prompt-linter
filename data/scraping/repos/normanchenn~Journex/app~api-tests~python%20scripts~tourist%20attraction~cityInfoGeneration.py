import cohere
import openai
import time
import json
import sys
from cohere.responses.classify import Example
co = cohere.Client('8GEb0w8gJW7TnKs6FLR45yWG0KSLIJNpeuIikuVF')
openai.api_key = "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"

countries = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user",
         "content": "Tell me all the countries in the world.",}
    ]
)
countryResponse = countries.choices[0].message.content
list = [country.strip() for country in countryResponse.split(",\n") if country.strip()]

for item in list:
    print(item)

# del list[0]
# for item in list:
#     tempPrompt = "List me the top 10 tourist attractions in " + item + " without description"
#     print(tempPrompt)
#     tempResponse = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "user",
#             "content": tempPrompt,}
#         ]
#     )
#     content = tempResponse.choices[0].message.content
#     items = content.split("\n")

#     f = open("testOutput.txt", "a")
#     f.write(item + "\n")
#     for outItem in items:
#         f.write(outItem + "\n")
#     f.write("\n")
#     f.close()
#     time.sleep(20)

#     # with open('touristPerCountry.txt', 'a') as f:
#     #     for outItem in items:
#     #         f.write(outItem)
#     #     f.write("\n")
