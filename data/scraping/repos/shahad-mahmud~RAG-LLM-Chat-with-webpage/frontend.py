import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import qdrant
from qdrant_client import QdrantClient

import src


def main():
    st.title("Ask your web page")

    if "links" not in st.session_state:
        st.session_state["links"] = []

    embedder = src.embeddings.get_hf_embedder()

    client = QdrantClient()
    vectorstore = qdrant.Qdrant(
        client, collection_name="web_pages", embeddings=embedder
    )

    llm = src.llms.get_llama2_chat()

    with st.sidebar:
        option = st.selectbox("Which LLM do you want to use?", ("LLaMA 2", "LLaVA"))

        if option == "LLaMA 2":
            llm = src.llms.get_llama2_chat()
        elif option == "LLaVA":
            llm = src.llms.get_llava_chat()
        else:
            raise ValueError(f"Invalid option: {option}")

        st.text_input(
            "web page links",
            key="web_page_link_add",
            type="default",
            placeholder="https://",
        )

        if st.session_state.web_page_link_add:
            link = st.session_state.web_page_link_add
            if link and link not in st.session_state.links:
                st.session_state.links.append(link)
                text = src.data.load_text(link, split=True)

                # add the text to the vectorstore
                vectorstore.add_documents(text)

        # show the added links
        for i, link in enumerate(st.session_state.links):
            if link:
                st.write(f"{i+1}. {link}")

    # chain = RetrievalQA.from_llm(llm=llm, retriver=vectorstore.as_retriever())
    chain = RetrievalQAWithSourcesChain(llm=llm, retriver=vectorstore.as_retriever())

    if question := st.chat_input(placeholder="Ask your web page"):
        st.chat_message("user").write(question)

        result = chain(question)

        st.chat_message("bot").write(result["answer"])


if __name__ == "__main__":
    main()
