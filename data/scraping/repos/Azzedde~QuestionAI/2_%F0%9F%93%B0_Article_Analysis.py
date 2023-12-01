import streamlit as st 
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter 
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv

chunk_size = 3000
chunk_overlap = 200

st.set_page_config(page_title="Article Analysis", page_icon="📰")

with st.sidebar:
    st.markdown('''
    ### 📰 Article Analysis: Understand the essence of articles without the fluff.
- paste an article link from the web
- get a brief summary of your article
- enjoy and support me with a star on [Github](https://www.github.com/Azzedde)
                
    ''')

load_dotenv()

def main():
    
    st.header("Summarize Articles from the Web 🌎")
    article_url = st.text_input("Enter the URL of the article you want to summarize")
    urls = []
    urls.append(article_url)
    if article_url != "":
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_text(data[0].page_content)
        docs = [Document(page_content=text) for text in texts[:]]
        llm = OpenAI(temperature=0)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        response = chain.run(docs)
        st.write(response)

if __name__ == "__main__":
    main()


