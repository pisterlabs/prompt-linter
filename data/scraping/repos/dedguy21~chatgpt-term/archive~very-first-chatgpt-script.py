#!/usr/bin/env python
import openai
import sys
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4001,
        n=1,
        stop=None,
        temperature=0.3
    )
    return response['choices'][0]['text']

while True:
    prompt = ""
    print(
        "\033[95mEnter your query up to 5000 characters\033[0m (Press 'Ctrl + D' to submit): ")
    try:
        while True:
            line = input()
            prompt += line + '\n'
    except EOFError:
        print("\033[1;34;40mPrompt accepted.\033[0m Awaiting response...")
        response = generate_response(prompt)
        print("\033[1;33;40mOpenAI response:\033[0m")
        print(response)
        print("\n")
    except KeyboardInterrupt:
        print("\033[1;31;40mExiting program.\033[0m")
        sys.exit()

''' below is the code for gpt-3.5-turbo-16k --- uncomment to use

        import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-16k",
  messages=[
    {
      "role": "user",
      "content": ""
    }
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
   '''
