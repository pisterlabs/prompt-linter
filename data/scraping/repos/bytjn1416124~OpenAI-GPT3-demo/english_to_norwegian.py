# -*- coding: utf-8 -*-
import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
from generalGpt3 import GPT3


prompt= '''English: I do not speak Norwegian.
Norsk: Jeg snakker ikke norsk.

English: See you later!
Norsk: Snakkes!

English: Where is a good restaurant?
Norsk: Hvor kan jeg finne en god restaurant?

English: What rooms do you have available?
Norsk: Hvilke rom er ledige?

English: What is the time?
Norsk: Hva er klokken?

English: '''

GPT3(prompt, "English:", "Norsk:",temperature=0.5,top_p = 1,frequency_penalty=0.0,presence_penalty=0.0)
