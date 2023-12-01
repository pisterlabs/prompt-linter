import streamlit as st
import openai 
import pinecone
st.title("Answers to life from the Gita")

st.text(" ")
st.image("https://i0.wp.com/telugubullet.com/wp-content/uploads/2017/11/3When-Arjuna-questioned-Lor.jpg")
st.text("             हरे कृष्‍णा, हरे राम|      ")

OPENAI_API_KEY = st.secrets['openai_key']
openai.api_key = OPENAI_API_KEY

index_name = "test" #name of the pinecone index
from langchain.vectorstores import Pinecone
text_field = "text" 
PINECONE_API_KEY = st.secrets['pinecone_key']
PINECONE_API_ENV = st.secrets['pinecone_env']

from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key= OPENAI_API_KEY
)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "test"

index = pinecone.Index(index_name)
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How to stay calm under pressure?"):
    documents = vectorstore.similarity_search(
    prompt,  # our search query
    k=7  # return 7 most relevant docs
)
    user_content = f"""
            Question:{prompt}

            Supporting Texts:
            Supporting Text 1: {documents[0].page_content}
            Supporting Text 2: {documents[1].page_content}
            Supporting Text 3: {documents[2].page_content}
            Supporting Text 4: {documents[3].page_content}
            Supporting Text 5: {documents[4].page_content}
            Supporting Text 6: {documents[5].page_content}
            Supporting Text 7: {documents[6].page_content}
            """

    system_content = "You are given a question and a pair of supporting passages from the Hindu sacred text BhagvadGita, which you can use as context to answer the question. If you don't know the answer, just say that you don't know. Don't try to make up an answer. Use the supporting text's example and support it with teachings from the answer. If you can't find the answer within the supporting texts, you can use your own knowledge of Bhagvad Gita and provide an answer. This is important that we don't leave the user without an answer"

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages = [{"role": "system", "content": f"{system_content}"},
                {"role": "assistant", "content": f"{user_content}"}],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
        message_placeholder.markdown(full_response)
        # st.write(f"""
        #     Question:{prompt}

        #     Supporting Texts:
        #     Supporting Text 1: {documents[0].page_content}
            
        #     Supporting Text 2: {documents[1].page_content}
            
        #     Supporting Text 3: {documents[2].page_content}
            
        #     Supporting Text 4: {documents[3].page_content}
            
        #     Supporting Text 5: {documents[4].page_content}
            
        #     Supporting Text 6: {documents[5].page_content}
            
        #     Supporting Text 7: {documents[6].page_content}
        # #     """
        #                             )
    st.session_state.messages.append({"role": "assistant", "content": full_response})


