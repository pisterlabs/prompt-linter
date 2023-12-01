import streamlit as st
import json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import os
import numpy as np

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def load_data(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

def get_agenda_items(recipe_name, data):
    for category, recipes in data.items():
        for recipe in recipes:
            if recipe['Recipe'] == recipe_name:
                return recipe['Agenda Items']
    return None

def main():
    tasks_data = load_data('tasks.json')
    flow_data = load_data('flow.json')

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4", temperature=.2, openai_api_key=openai_api_key)
    
    # Convert the tasks to Document objects
    documents = [Document(page_content=task) for task in tasks_data['tasks']]

    # Create an embeddings model
    embeddings = OpenAIEmbeddings()

    # Create a FAISS vectorstore from the documents
    db = FAISS.from_documents(documents, embeddings)

    st.title("Luma AI")
    recipe_name = st.text_input('Enter a task')
    if recipe_name:
        # Perform a similarity search
        similar_docs = db.similarity_search(recipe_name, k=1)
        if similar_docs:
            closest_task = similar_docs[0].page_content
            similarity = np.linalg.norm(np.array(embeddings.embed_query(recipe_name)) - np.array(embeddings.embed_query(closest_task)))
            agenda_items = get_agenda_items(closest_task, flow_data)
            if agenda_items:
                # Create a chain that uses the language model to generate a complete sentence
                template = "Based on your input, I suggest you to follow these steps: {agenda_items}. This suggestion is based on the recipe '{recipe_name}', which is {similarity}% similar to your input. The original recipe that it is matching with is '{closest_task}'."
                prompt = PromptTemplate(template=template, input_variables=["agenda_items", "recipe_name", "similarity", "closest_task"])
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                response = llm_chain.run({"agenda_items": ', '.join(agenda_items), "recipe_name": recipe_name, "similarity": round(similarity * 100, 2), "closest_task": closest_task})
                st.write(response)
                with st.expander("Details"):
                    st.write(f"Closest Luma Task: {closest_task}")
                    st.write(f"{similarity}% similar to that task")
            else:
                st.write('Agenda Items not found for the task')
        else:
            st.write('Task not found')

if __name__ == "__main__":
    main()