import weaviate
import os
import dotenv
import json
import openai
from decouple import config

dotenv.load_dotenv()
openai.api_key = config("OPENAI_API_KEY")
OPEN_API_KEY = os.getenv('OPENAI_API_KEY')

print("opening weaviate")
weaviate_class_name = "Corrected_Milken_Institute_data"

WEAVIATE_URL = "http://206.189.199.72:8080/"
client = weaviate.Client(
    url=WEAVIATE_URL,  # Replace with your endpoint
    additional_headers={
        "X-OpenAI-Api-Key": OPEN_API_KEY,
    }
)

print(client.schema.get())


def search_items(class_name, variables=[""], text_query="", k=10):
    results = client.query.get(class_name=class_name, properties=variables).with_near_text(
        {"concepts": text_query}).with_limit(k).do()
    return results["data"]["Get"][class_name]


def get_answer_stream(question: str):
    context = search_items(class_name=weaviate_class_name, variables=[
        "page_text"], text_query=question, k=5)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful and honest assistant that will chat with a user about the Milken Institute. In addition to your general knowledge, you have recently learned the following information: " + str(context)},
                  {"role": "user", "content": "This is my question: " + question}],
        max_tokens=2500,
        temperature=0.3,
        stream=True,
    )
    for part in response:
        if 'content' in part['choices'][0]['delta']:
            yield part['choices'][0]['delta']['content']


def get_openai_summary(text: str, question: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Your job is to describe a webpage taken from the Milken Institute website in 7 words or less. This webpage may or may not contain information relevant to a question that the user asked. Describe the webpage in under 7 words regardless. " +
                   "This is a piece of text from the webpage: " + text + " And this is the question the user asked: " + question}, ],
        max_tokens=50,
        temperature=0.3,
        stream=False,
    )
    print(response)
    return response["choices"][0]["message"]["content"]
