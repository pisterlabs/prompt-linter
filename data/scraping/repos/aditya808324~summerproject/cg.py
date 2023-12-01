#!/usr/bin/python3
import cgi
import json
import openai
import time
import cgitb

# Set the appropriate headers for a CGI script
print("Content-Type: text/html")
cgitb.enable()
print()

# Get the user's message from the request
form = cgi.FieldStorage()
user_message = form.getvalue('message')
openai.api_key = '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'
# Generate response using OpenAI GPT-3
response = openai.Completion.create(
    engine='text-davinci-003',
    prompt=user_message,
    max_tokens=50,
    temperature=1.2,
    n=1,
    stop=None
)
print(response.choices[0].text.strip())