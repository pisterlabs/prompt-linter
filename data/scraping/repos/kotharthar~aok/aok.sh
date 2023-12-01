#!/usr/bin/env python
import openai
import os
import sys
from colorama import init, Fore, Style, Back

# OpenAI Parameters
openai.api_key = os.environ["OPENAI_API_KEY"] # replace with your API key

# Max number of token to generate.  # Keep it brief for commandline chat case.
max_tokens = 512

# Max number of chat history to feed back. If this number is too large, it will cost more input Tokens
max_history_size = 4

# Initial framing system prompot
system_prompt = {"role": "system", "content":'For questions:\n Containing "command for/to" or "how to", provide an example command with a brief explanation in 5 bullets or less, each under 40 words.\n With "what is" or "explain", give a concise answer under 20 words, followed by a 75-word paragraph. For "explain" use additional words if necessary.\n Needing "code example" or "sample code", present example code in the specified language, or Python if not specified, plus an explanation in bullet points.\n Asking for "step by step" or "steps", detail steps in numbered bullets, each under 25 words.\n E: Requiring "summary" or "summarize" or for undefined instructions, create a 50-word key summary sentence, with context summaries in bullets under 25 words.\n \n Strictly use given format.\n """\n -------- Code --------\n \n {Command/Code}\n \n ----- Explanation -----\n \n Bullet 1\n """\n For last 2 format use the following format \n """\n ---- [Steps | Summary] ----\n {Output}\n """ '}

# keep_history function keeps only the last 4 messages
# in history by keeping first in first out approach.
history = []
def keep_history(message):
  if len(history) == max_history_size:
      history.pop(0)
  history.append(message)

# Check the commandline paramters.
# If the first param is `chat` then start the chat.
if (len(sys.argv) > 1 and sys.argv[1:][0] == "chat"):
  print("All " + chr(0x1F44C) + " !")
  while True:
    user_input = input("\n" + Fore.BLUE + "ME: " + Style.RESET_ALL)

    # If user_input is bye or exit exit the loop
    if (user_input in ["bye", "exit"]):
      print(Back.GREEN + Fore.RED + "AI:" + Style.RESET_ALL + " All" + chr(0x1F44C))
      break

    # Prepare the input message
    user_message = {"role": "user", "content": user_input}

    # Call to API REF: https://platform.openai.com/docs/api-reference/chat/create
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", # this is fixed for this ChatCompletion call
      max_tokens=max_tokens, # number of tokens to generate
      n=1,                   #  number of completions to generate
      temperature=1.0,       # Keep it close to zero for definite answer
      messages=([system_prompt] + history + [user_message]),
    )
    
    ai_message = response.choices[0].message # The reponse message from AI
    content = ai_message["content"]          #  The content of the AI's reponse message
    keep_history(user_message)      # Keep the last user message in history
    keep_history(ai_message)        # Keep the last AI message in history
    print(Back.GREEN + Fore.RED + "AI:" + Style.RESET_ALL, content)
    print('-' * 40)  
elif (len(sys.argv) > 1 and sys.argv[1:][0] == "raw"):
    the_input = sys.stdin.readlines()
    user_input = " ".join(the_input)  # All string
    user_input = user_input.replace('"', '\\"') # Escape
    user_input = user_input.replace('?', '\\?') # Escape
    user_message = {"role": "user", "content": user_input}
    response = openai.ChatCompletion.create(
       model="gpt-4", # this is fixed for this ChatCompletion call
       max_tokens=max_tokens*2, # number of tokens to generate
       n=1,                   #  number of completions to generate
       temperature=1.0,       # Keep it close to 1 for more creative answers.
       messages=([{"role":"system", "content": "Act as a helpful smart bot. Answer the given question is brief and most smartest way possible."}] + [user_message]),
    )
    print("\n" + response.choices[0].message["content"])
else:
    the_input = sys.argv[1:]
    # if The_input is empty read the text from STDIN
    if(the_input == []):
      the_input = sys.stdin.readlines()
    user_input = " ".join(the_input)  # All string
    user_input = user_input.replace('"', '\\"') # Escape
    user_input = user_input.replace('?', '\\?') # Escape
    user_message = {"role": "user", "content": user_input}
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", # this is fixed for this ChatCompletion call
      max_tokens=max_tokens*2, # number of tokens to generate
      n=1,                   #  number of completions to generate
      temperature=1.0,       # Keep it close to 1 for more creative answers.
      messages=([system_prompt] + [user_message]),
    )
    print("\n" + response.choices[0].message["content"])
