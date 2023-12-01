import json

import openai
import requests
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient

from aidevs_other_stuff.qdrantHelper import QdrantHelper
from helper import Helper
from aidevs_other_stuff.jsonReader import JsonReader

NBP_CURRENCY_API = "https://api.nbp.pl/api/exchangerates/tables/a/?format=json"
COUNTRY_API = "https://restcountries.com/v3.1/all"

# C0401 - knowledge
# Automat zada Ci losowe pytanie na temat kursu walut, populacji wybranego kraju lub wiedzy ogólnej.
# Twoim zadaniem jest wybór odpowiedniego narzędzia do udzielenia odpowiedzi (API z wiedzą lub skorzystanie z wiedzy modelu).
# W treści zadania uzyskanego przez API, zawarte są dwa API, które mogą być dla Ciebie użyteczne.
def fallback_with_vectordb(currency, cache_name, results):
    print("Fallback with vector db")
    ##load
    MEMORY_PATH = "knowledge_task_vectordb_init.json"
    with open(MEMORY_PATH, 'w') as f: json.dump(results, f)
    QdrantHelper.init_collection(results, collection_name=cache_name, limit=1500)
    ##do similarity search
    embeddings = OpenAIEmbeddings()
    question_embedding = embeddings.embed_query(currency)

    qdrant = QdrantClient("http://localhost:6333")
    search = qdrant.search(cache_name, query_vector=question_embedding, limit=1, query_filter={
        'must': [
            {
                'key': 'source',
                'match': {
                    'value': cache_name
                }
            }
        ]
    })
    found_entry = search[0].payload
    information_bout_it = found_entry['content']
    print("Found information ", information_bout_it)
    return information_bout_it


def search_currency_api(currency_i_hope):
    print("Searching the NBP currency API for: " + currency_i_hope)
    currency = currency_i_hope.lower()
    results = requests.get(NBP_CURRENCY_API)
    answer = None
    for result in results.json()[0]["rates"]:
        if result["currency"] == currency:
            answer = result["mid"]
            break
        # could add search local vector db if not found the above if needed as a fallback
    if(answer == None):
        fallback_with_vectordb(currency, "NBP_CURRENCY_API_CACHE", results.json()[0]["rates"])
        answer = result["mid"]
    return answer

def search_country_api(country_i_hope):
    print("Searching the country API for: " + country_i_hope)
    country = country_i_hope
    print("Country: " + country)
    answer = None
    results = requests.get(COUNTRY_API)
    for result in results.json():
        if result["name"]["common"] == country:
            answer = result["population"]
            break
        # could add search local vector db if not found the above if needed as a fallback
    if(answer == None):
        result = fallback_with_vectordb(country, "COUNTRY_API_CACHE", results.json())
        answer = result["population"]
    return answer


def ask_the_model(question):
    print("asking the openai for: " + question)
    openai.api_key = Helper().get_openapi_key()
    ai_resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "write separate paragraphs for each title, one short paragraph for each title"},
            {"role": "user",
             "content": question}
        ])
    ai_response = ai_resp.choices[0].message.content
    print(ai_response)
    return ai_response


class Knowledge:
    @staticmethod
    def generate_answer(test_data):
        question = str(test_data.json()["question"])
        #question = str(test_data.json()["justdie"]) # if i want it to die -> die die die

        result = None
        if question.__contains__("kurs") or question.__contains__("exchange"):
            return search_currency_api(question.split()[-1])
        elif question.__contains__("population") or question.__contains__("populacja"):
            return search_country_api(question.split()[-1])

        if result == None:
            return ask_the_model(question)


if __name__ == '__main__':
    test_data = Helper.create_simulated_response(
        b'{"question":"Whats the current exchange rate of eur", "justdie":"Whats the current exchange rate of eur"}')
    test_data = Helper.create_simulated_response(
        b'{"question":"Whats the population of iranu", "justdie":"Whats the population of Iranu"}')
    # test_data = Helper.create_simulated_response(b'{"question":"Is the sky red", "justdie":"Is the sky red"}')

    ans = Knowledge().generate_answer(test_data)
    print(ans)
