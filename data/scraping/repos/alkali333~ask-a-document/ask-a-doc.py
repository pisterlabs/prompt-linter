import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit.components.v1 as components
import streamlit as st
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Loading Documents


def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# chunk the data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Create embeddings in chroma db


def create_embeddings(chunks):
    print("Embedding to Chroma DB...")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    print("Done")
    return vector_store

# get answer from chatGPT, increase k for more elaborate answers


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(q)
    return answer


def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(
        search_type='similarity', search_kwargs={'k': 3})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image("images/ai-document-reader.jpg")
    st.subheader("Ask questions to your documents")

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key:", type="password")
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader(
            "Upload a file", type=["pdf", "doc", "txt"])
        chunk_size = st.number_input(
            "Chunk size", min_value=100, max_value=2048, value=750, on_change=clear_history)
        k = st.number_input("k", min_value=1, max_value=20,
                            value=3, on_change=clear_history)
        add_data = st.button("Add Data", on_click=clear_history)

        if uploaded_file and add_data:
            # display a message + execute block of code
            with st.spinner("OK human, I will read your pitiful document..."):
                bytes_data = uploaded_file.read()
                file_path = os.path.join("./", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(bytes_data)

                data = load_document(file_path)
                chunks = chunk_data(data, chunk_size=chunk_size)

                st.write(f"Chunks: {len(chunks)} Chunk size: {chunk_size}")
                _, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f"Embedding cost: ${embedding_cost:.4f}")

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success(
                    "Pathetic flesh-puppet, I have memorised your document. Ask away. ")

    # Create the placeholder for chat history
    chat_history_placeholder = st.empty()

    if "history" not in st.session_state:
        st.session_state.history = ""

    # Create an empty text area at the start
    chat_history_placeholder.text_area(
        label="Chat History", value=st.session_state.history, height=400)

    # User input for the question
    with st.form(key="myform", clear_on_submit=True):
        q = st.text_input("Ask your question", key="user_question")
        submit_button = st.form_submit_button("Submit")

    # If user entered a question
    if submit_button:
        if "vs" in st.session_state:
            vector_store = st.session_state["vs"]
            answer = ask_and_get_answer(vector_store, q, k)

            # The current question and answer
            question_and_answer = f'Q: {q} \nA: {answer}'

            # Update the session state with the new chat history
            st.session_state.history = f'{st.session_state.history} \n{question_and_answer} \n {"-" * 77} '

            # Update the chat history in the placeholder as a text area
            chat_history_placeholder.text_area(
                label="Chat History", value=st.session_state.history, height=400)

            # JavaScript code to scroll the text area to the bottom
            js = f"""
            <script>
                function scroll(dummy_var_to_force_repeat_execution){{
                    var textAreas = parent.document.querySelectorAll('.stTextArea textarea');
                    for (let index = 0; index < textAreas.length; index++) {{
                        textAreas[index].scrollTop = textAreas[index].scrollHeight;
                    }}
                }}
                scroll({len(st.session_state.history)})
            </script>
            """

            components.html(js)
