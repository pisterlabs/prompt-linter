# Summarize articles using the chatgpt api 
import os
import openai

# Get key from openaikey.txt
# NB: Do not commit this file to git! 
with open('../openaikey.txt', 'r') as file:
    openai.api_key = file.read()

advices = []
# Loop over the texts in the data folder
# For each text, summarize it using the chatgpt api
# Add the points to the advices list
for filename in os.listdir('data'):
    with open('data/' + filename, 'r') as file:
        text = file.read()
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Vat deze tekst samen in drie bulletpoints (in het Nederlands), in de vorm van adviezen: {text}"}
            ]
        )
        points = completion.choices[0].message.content
        print(points)
        advices.append(points)

# Write the advices to a file
with open('advices.txt', 'w') as file:
    file.write('\n'.join(advices))
