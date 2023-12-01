import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


with open('eng_to_elv.json') as file:
    sindarin_pairs = json.load(file)

sindarin_dict = {}
for pair in sindarin_pairs:
    for key, value in pair.items():
        sindarin_dict[key] = value
        
def find_sindarin_pair(word):
    return sindarin_dict.get(word, ' ')

role = """You are Elanorion, a wise Elf scholar, who spend her life learning all the languages of men. You posses immense knowledge of the elvish language Sindarin and the human language of English. Today your task is to help a human, he has an old Sindarin dictionary where he can find translations for some words but he needs your help to form sentences. He will provide you a sentence in english, alongside all the translation pairs he could find. It is up to you to use your knowledge of Sindarian grammar to help the human trasnlate the sentence in it's entirety, as truthfully to the original as possible.


The human will give information as such : 
"Example sentence" 
words pairs eg. "friend":"mellon"

You will respond with the translation of the original text in Sindarin.

"""


user_input = input().replace(",","")
words = user_input.split(" ")
translation_pairs = ""

sindarin = ""


for word in words :
    sindarin_pair = find_sindarin_pair(word)
    if (find_sindarin_pair(word)!=" ") :
        translation_pairs += word + ":" + sindarin_pair + "\n"

        sindarin += sindarin_pair +" "


toks = int(len(sindarin) * 1.5)

evaluation = openai.Completion.create (

    engine = "text-davinci-003",
    prompt = role + user_input + translation_pairs,
    max_tokens = toks,
    n=1,
    temperature = 0.5

)
print(translation_pairs)
print (evaluation.choices[0].text.strip())