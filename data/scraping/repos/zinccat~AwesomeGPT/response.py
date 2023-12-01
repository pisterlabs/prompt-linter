from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains.question_answering import load_qa_chain
import pinecone
import os
from key import PINECONE_API_KEY, PINECONE_API_ENV, OPENAI_API_KEY, PINECONE_INDEX_NAME
import openai

openai.api_key = OPENAI_API_KEY

# using gpt-3.5 for now
def gpt(messages, model="gpt-3.5-turbo", temperature=0.3):
    output = openai.ChatCompletion.create(
    model=model,
    messages=messages,
        temperature=temperature
    )
    return output.choices[0]['message']['content']

system_prompt = """You are a nice and helpful assistant who is expert of AI. Given the following verified sources and a question, create a precise answer in markdown. For example, if there are names in the sources, you should provide it. If you can't answer the question based on the given information, please say 'I don't have enough information on this topic'.

Sources: {}

The question is: """

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    messages = [
            {"role": "system", "content": ""},
        ]
    while True:
        question = input("Input your question: ")
        docs = vectorstore.similarity_search(question, k=2)
        sources = '\n\n'.join([doc.page_content for doc in docs])
        # print(sources)
        print('\n')
        prompt = system_prompt.format(sources)
        messages[0]['content'] = prompt
        # maybe consider erase part of the history to save cost and fit the context window constraint.
        messages.append({"role": "user", "content": question})
        response = gpt(messages)
        print("Response:", response)
        print('\n')
        messages.append({"role": "assistant", "content": response})