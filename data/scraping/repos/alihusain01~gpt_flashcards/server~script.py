from PyPDF2 import PdfReader
import os
import openai
import authentication
import json
import sys

def parse_pdf():
    reader = PdfReader("uploads/" + sys.argv[1])
    # reader = PdfReader("../API/HY3238_20230314_Cold War.pdf")
    number_of_pages = len(reader.pages)
    text = ""

    for i in range(3, 5): # changed the number of pages for faster testing
        page = reader.pages[i]
        text += page.extract_text()

    # print("Text is converted")

    return text

def make_response(prompt):
  openai.api_key = authentication.API_KEY

#   print("Sending API call to GPT")

  response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages = [
                        {"role": "user", "content": "Make 15 flashcards from the following text in the form of a json string using key values term and definition" + prompt + "\n"}
                        ],
              temperature = 0
          )

  return response.choices[0].message.content

my_dict = make_response(parse_pdf())
json_dict = json.dumps(my_dict)

print(json_dict)
