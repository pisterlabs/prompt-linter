import streamlit as st
from streamlit_chat import message
from langchain import OpenAI
#from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain import PromptTemplate
from langchain import LLMChain
import openai
import pinecone
import os
from dotenv import load_dotenv

load_dotenv('../.env')

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_API_ENV = os.environ['PINECONE_API_ENV']
PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']
EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL']

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV  # may be different, check at app.pinecone.io
)
# connect to index
index = pinecone.Index(PINECONE_INDEX_NAME)

@st.cache_resource
def LLM_chain_response():
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="Answer the question based on the context below. If you cannot answer based on the context, or general knowledge of the company Wells Fargo, truthfull answer that you don't know. Use Markdown and text formatting to format your answer. \n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
    )

    llm = OpenAI(
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY,
        model_name="text-davinci-003",
        max_tokens=128
    )

    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory= ConversationSummaryBufferMemory(llm=llm, max_token_limit=256),
        verbose=True
    )
    return chatgpt_chain

# Define the retrieve function
def retrieve(query):
    # retrieve from Pinecone
    res = openai.Embedding.create(input=[query],model=EMBEDDING_MODEL)
    xq = res['data'][0]['embedding']

    # get relevant contexts
    pinecone_res = index.query(xq, top_k=10, include_metadata=True)
    contexts = [x['metadata']['chunk_text'] for x in pinecone_res['matches']]

    pinecone_contexts = (
        "\n\n---\n\n".join(contexts)
    )
    return pinecone_contexts

# From here down is all the StreamLit UI.
image = open("Pinecone logo white.png", "rb").read()
st.image(image)
st.write("### Lord of the Rings - Fellowship of the Ring Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Who is Bilbo Baggins?", key="input")
    return input_text

# Main function for the Streamlit app
def main():
    chatgpt_chain = LLM_chain_response()
    user_input = get_text()
    if user_input:
        with st.spinner("Thinking..."):
            query = user_input
            pinecone_contexts = retrieve(query)
            output = chatgpt_chain.predict(input=query + '\nContext: ' + pinecone_contexts)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="shapes")

if __name__ == "__main__":
    main()
