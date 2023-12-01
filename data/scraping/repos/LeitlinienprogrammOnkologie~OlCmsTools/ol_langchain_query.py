from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import pinecone
import openai
from IPython.display import display


openai.api_key = "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"

pinecone.init(api_key="a5b5539f-8e83-4b74-ab9f-aa4a1da90fd9", environment="us-west1-gcp")
index = pinecone.Index("test-index")

while True:
    user_message = input("Frage: ")
    if user_message.lower() == "quit":
            break
    xq = openai.Embedding.create(input=[user_message], engine="text-embedding-ada-002")['data'][0]['embedding']
    query_response = index.query([xq], top_k=5, include_metadata=True)
    matches = query_response['matches']

    contexts = [x['metadata']['text'] for x in query_response['matches']]
    augmented_query = "Context:\"".join(contexts) + "\" " + user_message

    system_msg = "You are a helpul machine learning assistant and tutor. Answer questions based exclusively on the context provided without any additional queries, or say I don't know."

    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": augmented_query}])

    display(chat["choices"][0]['message']['content'])