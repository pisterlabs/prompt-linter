import chromadb
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")


def read_and_embed_file(file_name, title):
    with open(file_name, 'r') as file:
        data = file.read()

    # split into paragraphs
    paragraphs = data.split('\n\n')
    metadata_list = []
    ids_list = []

    for idx, paragraph in enumerate(paragraphs):
        metadata_list.append({"source": f"{title}"})
        ids_list.append(f"id{idx + 1}")

    collection.add(
        documents=paragraphs,
        metadatas=metadata_list,
        ids=ids_list
    )


def generate_prompt(sources, question):
    return f"""
    
    Write a paragraph, addressing the question, and combine the text below to obtain relevant information. Cite sources using in-text citations with square brackets.

    For example: [1] refers to source 1 and [2] refers to source 2. Cite once per sentence.

    If the context doesn't answer the question. Output "I don't know".

    {sources}

    Question: {question}
    Result:"""


def make_openai_call(context, question):
    sources = ''
    for idx, paragraph in enumerate(context):
        sources += f"Source {idx + 1}: {paragraph}\n"

    prompt = generate_prompt(sources, question)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # or gpt-4
        messages=[
                {"role": "system", "content": "You are HonestGPT. Make sure all your answers cite the sources you used as in-text citations."},
                {"role": "user", "content": f"{prompt}"},
            ]
    )

    return response['choices'][0]['message']['content']


def pretty_print_results(query, summary, sources):
    print(query)
    print("--------------------")
    print(summary)

    print("*********")
    print("Sources:")
    for idx, source in enumerate(sources):
        print(f"{idx + 1}: {source}")
    print("*********")


if __name__ == "__main__":
    # read text and embed file
    read_and_embed_file('texts/coffee.txt', "coffee-article")
    read_and_embed_file('texts/state_of_the_union.txt', "state-of-the-union")

    query = 'What is the difference between a cappuccino and a latte?'
    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    # Another example:

    # query = 'What did Biden say about Justice Breyer?'
    # results = collection.query(
    #     query_texts=[query],
    #     n_results=5
    # )

    top_results = results["documents"][0]

    pretty_print_results(query, make_openai_call(
        top_results, query).strip(), top_results)
