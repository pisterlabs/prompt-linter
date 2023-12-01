import pandas as pd
import openai

# Load the Excel sheet containing the acronyms and definitions
df = pd.read_excel('abbreviations.xlsx')

# Create a dictionary mapping acronyms to definitions
acronym_definitions = dict(zip(df['Acronym'], df['Definition']))
acronym_descriptions = dict(zip(df['Acronym'], df['Description']))

# OpenAI API credentials
openai.api_key = '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'


# Define a function to get the definition of an acronym
def get_definition(acronym):
    return acronym_definitions.get(acronym)


def get_description(acronym):
    return acronym_descriptions.get(acronym)


# Define a function to generate a response using OpenAI
def generate_response(user_input):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=f"What does {user_input} mean?",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()


# Chatbot loop
def run_chatbot(user_input):
    answer = None
    flag = False

    user_input = user_input.upper().split(" ")

    for word in user_input:
        if word in acronym_definitions:
            response = ""
            definition = acronym_definitions.get(word)
            description = acronym_descriptions.get(word)

            if definition == '***':

                if description == '***':
                    no_description = "It is a valid acronym, but I don't know its definition or description. "
                    response += no_description
                    answer = f"{response}"
                    flag = True
                    break
                else:
                    gpt_response = generate_response(description)
                    response += gpt_response
                    answer =  f"This isn't an acronym but I still know it. {response}"
                flag = True
                break

            response += definition + '- '

            if description == '***':
                no_descrip = "Sorry, I don't have a description for this acronym."
                response += no_descrip
                answer =  f"{response}"
                flag = True
                break

            gpt_response = generate_response(description)
            response += gpt_response

            answer =  f"{response}"
            flag = True
            break

    if not flag:
        answer = f"Sorry, I have neither a definition nor description for this acronym."
    return answer