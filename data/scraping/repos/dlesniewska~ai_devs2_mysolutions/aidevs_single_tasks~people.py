import openai
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient

from helper import Helper


# Pobierz, a następnie zoptymalizuj odpowiednio pod swoje potrzeby bazę danych https://zadania.aidevs.pl/data/people.json
# Twoim zadaniem jest odpowiedź na pytanie zadane przez system. Uwaga!
# Pytanie losuje się za każdym razem na nowo, gdy odwołujesz się do /task.
# Spraw, aby Twoje rozwiązanie działało za każdym razem, a także, aby zużywało możliwie mało tokenów.
# Zastanów się, czy wszystkie operacje muszą być wykonywane przez LLM-a - może warto zachować jakiś balans między światem kodu i AI?

class People:
    @staticmethod
    def generate_answer(test_data):
        collection_name = "ai_devs_people_task"
        question = str(test_data.json()["question"])

        # create embeddings for sample text query
        embeddings = OpenAIEmbeddings()
        question_embedding = embeddings.embed_query(question)

        qdrant = QdrantClient("http://localhost:6333")
        search = qdrant.search(collection_name, query_vector=question_embedding, limit=1, query_filter={
            'must': [
                {
                    'key': 'source',
                    'match': {
                        'value': collection_name
                    }
                }
            ]
        })

        print("Search", search)
        print("Search content", search)
        found_entry = search[0].payload
        information_bout_person = str(found_entry['content'])
        print("Found information ", information_bout_person)

        openai.api_key = Helper().get_openapi_key()
        ai_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"##Facts: {information_bout_person}##"},
                {"role": "user",
                 "content": question}
            ])
        ai_result = ai_resp.choices[0].message.content
        return ai_result

