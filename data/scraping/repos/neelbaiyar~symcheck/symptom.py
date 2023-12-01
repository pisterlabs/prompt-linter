#pip install metaphor-python
import os
import requests
import openai
from metaphor_python import Metaphor
from bs4 import BeautifulSoup

# Initialize the Metaphor AI client with your API key
openai.api_key = "OPENAI-KEY-HERE"

metaphor = Metaphor("METAPHOR-AI-KEY-HERE")

user_input = input("Enter your symptoms: ")
nums = input("Number of results: ")


'''USER_QUESTION = "I wonder what I health concerns I have with symptoms of" + user_input + ":"

SYSTEM_MESSAGE = "You are a doctor that generates one or two word responses based on user questions. Only generate one search query."

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_QUESTION},
    ],
)


query = completion.choices[0].message.content
search_response = metaphor.search(
    query, use_autoprompt=True, start_published_date="2023-09-01"
)

print(f"URLs: {[result.url for result in search_response.results]}\n")

'''
response = metaphor.search(
  "What are some home remedies recipes that can help me alleviate" + user_input + ":",
  num_results=int(nums),
  use_autoprompt=True,
  start_published_date="2022-10-01"
)

contents_response = response.get_contents()

# Print content for each result
for content in contents_response.contents:
  print(f"Title: {content.title}\nURL: {content.url}\n")
    # Get the HTML content of the first result
  response = requests.get(content.url)
  html_content = response.content

  # Parse the HTML content
  soup = BeautifulSoup(html_content, "html.parser")
  
  print(soup.text)

  # Summarize the HTML content
  
  USER_QUESTION = soup.text
  SYSTEM_MESSAGE = "Can you please summarize this and output the remedies/solutions in this set of text"

  completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": SYSTEM_MESSAGE},
          {"role": "user", "content": USER_QUESTION},
      ],
  )
  
  # Extract the summary from the response
  summary = completion.choices[0].message["content"]

  # Print the summary
  print("Summary:")
  print(summary)
  
#You exceeded your current quota, please check your plan and billing details. -OpenAI
