"""
Name suggestion using openai API
"""

import os
import sys
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_prompt(gender,
                    pronoun,
                    last_name,
                    nationality,
                    mother_name,
                    father_name,
                    first_name_suggestion_1,
                    first_name_suggestion_2,
                    first_name_suggestion_3,
                    how_many_suggestions):

    return"""Find a first name for a baby {}, {} last name is {} and {} nationality is {}. 
    The mother's first name is {} and the father's first name is {}.
    I want {} first name to fit well {} last name.
    Names I like are: {}, {} and {} but don't suggest any of those names.
    Suggest me {} first names""".format(gender,
                          pronoun,
                          last_name.capitalize(),
                          pronoun,
                          nationality,
                          mother_name,
                          father_name,
                          pronoun,
                          pronoun,
                          first_name_suggestion_1.capitalize(),
                          first_name_suggestion_2.capitalize(),
                          first_name_suggestion_3.capitalize(),
                          how_many_suggestions)

gender = input("is it a boy or a girl ? ")

if gender == "boy":
    pronoun = "his"
elif gender == "girl":
    pronoun = "her"
else:
    pronoun = "it"

last_name = input("what will be {} last name? ".format(pronoun))
nationality = input("what will be {} nationality? ".format(pronoun))
mother_name = input("what is {} mother first name? ".format(pronoun))
father_name = input("what is {} father first name? ".format(pronoun))
first_name_suggestion_1 = input("what name do you like? ")
first_name_suggestion_2 = input("what name do you like? ")
first_name_suggestion_3 = input("what name do you like? ")

name_suggestion = input("Do you want me to suggest you some names? ")

if name_suggestion == "yes":
    how_many_suggestions = int(input("How many names do you want me to suggest? "))
    if how_many_suggestions > 10:
        how_many_suggestions = int(input("There is too many, choose less than ten, please "))
        if how_many_suggestions > 10:
            print("You didn't listen to me...Bye")
            sys.exit()
else:
    print("Bye")
    sys.exit()

print("thinking...")
temperature = float(input("Do you want me to be more creative or more pragmatic ? more or less temperature "))

response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=generate_prompt(gender,
                         pronoun,
                         last_name,
                         nationality,
                         mother_name,
                         father_name,
                         first_name_suggestion_1,
                         first_name_suggestion_2,
                         first_name_suggestion_3,
                         how_many_suggestions),
    temperature=temperature,
    max_tokens=2000,
    top_p=1,
    best_of=1,
    frequency_penalty=0,
    presence_penalty=0
)
print(response.choices[0].text)
