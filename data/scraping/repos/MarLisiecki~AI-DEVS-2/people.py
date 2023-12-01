import json
import os
import re
from typing import List

import qdrant_client
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant

from solver.solver import Solver

URL_FOR_DATA = "https://zadania.aidevs.pl/data/people.json"


def get_data(url: str) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    with open("people.json", mode="w", encoding="utf-8") as file:
        content = json.dumps(response.json())
        file.write(content)


def get_name_surname(question: str) -> List[str]:
    person = re.findall(r"\b[A-Z][a-z]*\b", question)
    return person  # [name, surname]


def temp_qdrant_handler(question):
    client = qdrant_client.QdrantClient(
        url="http://localhost:6333/",
    )
    os.environ["QDRANT_COLLECTION"] = "people-task"

    # collection_config = qdrant_client.http.models.VectorParams(
    #     size=1536,  # 768 for instructor-xl, 1536 for OpenAI
    #     distance=qdrant_client.http.models.Distance.COSINE
    # )
    #
    # client.recreate_collection(
    #     collection_name=os.getenv("QDRANT_COLLECTION"),
    #     vectors_config=collection_config
    # )
    print(client.get_collections())
    embeddings = OpenAIEmbeddings()

    vectorstore = Qdrant(
        client=client, collection_name="people-task", embeddings=embeddings
    )

    def person_info(person_dict):
        """
        This function takes a dictionary representing a person's details and
        returns a formatted string with all the information.

        :param person_dict: A dictionary containing person's details.
        :return: A string with the person's information.
        """
        info = f"Name: {person_dict.get('imie', 'N/A')}\n"
        info += f"Surname: {person_dict.get('nazwisko', 'N/A')}\n"
        info += f"Age: {person_dict.get('wiek', 'N/A')}\n"
        info += f"About: {person_dict.get('o_mnie', 'N/A')}\n"
        info += f"Favorite Character from Kapitan Bomba: {person_dict.get('ulubiona_postac_z_kapitana_bomby', 'N/A')}\n"
        info += f"Favorite TV Show: {person_dict.get('ulubiony_serial', 'N/A')}\n"
        info += f"Favorite Movie: {person_dict.get('ulubiony_film', 'N/A')}\n"
        info += f"Favorite Color: {person_dict.get('ulubiony_kolor', 'N/A')}"

        return info

    # def get_chunks():
    #     chunks = []
    #     with open("people.json") as f:
    #         raw_text = json.loads(f.read())
    #         for element in raw_text:
    #             person = person_info(element)
    #             chunks.append(person)
    #
    #     return chunks
    #
    # # get_chunks()
    #
    # texts = get_chunks()

    # vectorstore.add_texts(texts)

    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    query = "Answer the question: {question}".format(question=question)
    response = qa.run(query)

    return response


def people(input_data: dict) -> dict:
    question = input_data["question"]
    res = question[0].lower() + question[1:]
    get_data(URL_FOR_DATA)
    # name, surname = get_name_surname(res)

    # print(input_data)
    print(question)
    answer = temp_qdrant_handler(question)
    prepared_answer = {"answer": answer.strip()}
    print(answer)
    return prepared_answer


if __name__ == "__main__":
    # TODO REFACTOR
    sol = Solver("people")
    sol.solve(people)
