import argparse
import sys
import os
import openai


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from api_secrets import API_KEY
#from dtu_api import API_KEY

import numpy as np

openai.api_key = API_KEY


def divide_mcq(question):
    print("Opdeler Multiple Choice spørgsmål...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du omskriver multiple choice spørgsmål til 3 kortere og mere præcise spørgsmål, der hver fokuserer på et enkelt aspekt eller emne, men tilsammen dækker hele multiple choice spørgsmålet. Omskriv følgende multiple choice spørgsmål til 3 mere præcise spørsgsmål, og angiv dem således: Spørgsmål 1: [indsæt]\nSpørgsmål 2: [indsæt]\nSpørgsmål 3: [indsæt]."},
            {"role": "user", "content":  "I én af hovedteoriretningerne inden for budgetteori taler man nogen gange om “bureaushaping”. I en anden hovedteoriretning taler man nogen gange om “grønthøstermetode”. Hvilke to hovedteoriretninger er der tale om? a) Efterspørgselsorienteret og økologisk  budgetteori b) Inkrementel og udbudsorienteret budgetteori c) Udbudsorienteret og efterspørgselsorienteret budgetteori d) Inkrementel og efterspørgselsorienteret budgetteori"},
            {"role": "assistant", "content": "Spørgsmål 1: Hvad er budgetteori?\nSpørgsmål 2: Hvad er bureaushaping inden for budgetteori?\nSpørgsmål 3: Hvad er grønthøstermetode inden for budgetteori?"},
            {"role": "user", "content": question},

        ],
        temperature=0.7,
        max_tokens=500,
    )

    answer = response['choices'][0]['message']['content']
    subquestions = answer.split('\n')
    subquestions = [q.split(': ')[1] for q in subquestions if q]
    print("Genererede underspørgsmål: ")
    print('\n'.join(subquestions))
    return subquestions


if __name__ == "__main__":
    question = "Hvad er et ofte fremført argument for administrativ centralisering (frem for decentralisering)? a) Centralization speeds decision making by reducing the overload of information which otherwise clogs the upper reaches of a decentralized hierarchy b) Centralization encourages innovation c) Centralization improves staff motivation and identification d) Centralization makes the line of accountability clearer and more easily understood by citizens"
    transformed_questions = divide_mcq(question)
    #print(transformed_questions)
