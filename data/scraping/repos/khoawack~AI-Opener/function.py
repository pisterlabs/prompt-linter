import os
import openai
import convert_pdf_to_string

openai.organization = "ENTER ORG HERE"
openai.api_key = "ENTER API KEY HERE"

def compare(source, question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f'''Use strictly only this provided text to answer the question. Else respond with how that was not provided in the text. Also start by saying according to the text provided. You can interpret if a little information is missing but make sure you say that it is an interpretation.
    Source: [{source}]

    Question: [{question}]''',
        temperature=0.1,
        max_tokens=500,
    )



    return response.choices[0].text



# source =  convert_pdf_to_string.extract_text('sample3.pdf')
# print(source)
# question = "why was minecraft so popular compared to gta5"

# print(compare(source, question))
