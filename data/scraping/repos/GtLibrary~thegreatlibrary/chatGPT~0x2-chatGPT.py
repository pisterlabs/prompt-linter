#!/usr/bin/env python

"""chatGPT.py: Sample Python program which will act like a chatbot"""

__author__      = "Adithya Vinayak Ayyadurai;John R Raymond"


import os
import openai
import inspect

import dotenv
#dotenv.read_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
dotenv.read_dotenv("/home/john/bakerydemo/.env")
API_KEY = os.getenv("OPENAI_API_KEY")
#print("API_KEY: ", API_KEY)

# can be expanded as user wish

ESCAPE_KEYS = ["Exit"]

openai.api_key = API_KEY


def makeCall(message_arr):

  #thread_stub = {"role": "user", "content": "You are a helpful assistant."}
  thread_stub = {"role": "system", "content": "I am world famous author and programmer Donald Knuth, and you are my writing assistant. Weave my skills. :: Your job is to polish my text so it is ready to go to print. :: You are version Pi of Donald Knuth Edition of Vanity Printer[TM]"}
  thread_message = []
  thread_message.append(thread_stub)
  for val in message_arr:
    thread_message.append(val)

  #print("message array: ", message_arr)
  #print("thread message: ", thread_message)
  print(inspect.signature(openai.ChatCompletion.create))

  completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=thread_message,
          temperature=0.0)

  #completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=message_arr)
  return completion.choices[0].message


flag = True 

message_array = []

while flag:
  user_input = input("\nEnter the text: ")
  if user_input in ESCAPE_KEYS:
    flag = False
    continue

  user_input = """
Prologue: The Sins of Men
Only stories told in song live forever.
—From The Living Book of the Dead

The sun set hours ago leaving Gaz in shadows as he made his way home. Now only the ledge separated him from his bed. Ghosts were said to inhabit such cracks in Gearthrum’s crust. Here in the dark alone, Gaz was too afraid to disbelieve the stories. “Luckily, I’ve been smart,” he muttered to himself. He had spent his last coin to buy a whiff of tarsk to keep him company on the way home. Uncorking the bottle, he took a quick slug. “Hah!” He slammed the cork into the mouth of the jug. Emboldened, he took a step forward. 
A shadow suddenly loomed up in front of him. Gaz clutched the jug to his chest with both hands. “I see you,” the shadow said.
Gaz stepped backwards, putting his back against the wall of the cliff. “Who—who is it?” 
The moon briefly blinded him, but then he saw the figure was Rennly Rickot. “Don’t you recognize me?” Rennly asked with a smirk on his moonlit face.
Gaz stepped forward, farther out onto the ledge. “Rennly,” he said. “You’re back.” He pushed his small jug of tarsk towards Rennly’s dark outline.
“And I’m not just here for visiting either,” Rennly said, his arms folded in front of him. He seemed more serious now, but still the same Rennly as ever.
“I heard you were a censor,” Gaz said. He struggled not to slur his words. “And I know I have to do better—in front of others—but look at you, all in black.” Gaz wanted to be proud of Rennly, but he wasn’t his friend really. Was he? “Why are you here?” Gaz asked. “It’s the middle of nowhere.”
Rennly finally relaxed his stance. “To tidy up some loose ends.” 
A hand with dark spots, one Gaz knew too well, reached for the jug he held. Gaz snatched it back. “Let me—...get the cork.” He popped the top and took another swig himself. “Look at me, drinking with a censor.” He again held the jug out for Rennly. “Who would have thought it?”
Rennly snatched the jug from him but only sniffed the lip and did not drink. “Hmm,” Rennly grunted.
::Edit for spelling, grammar, voice, and style.::Rearange sentence flow: on::Lively language: on::
  """
  message_obj = {"role": "user", "content": user_input}
  message_array.append(message_obj)
  response_message = makeCall(message_array)
  message_array.append(response_message)
  print(message_array)
  print("\nSystem: %s" %(response_message['content']))
