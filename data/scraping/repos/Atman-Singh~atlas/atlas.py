from dotenv import load_dotenv
# import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import time
from requests_ratelimiter import LimiterSession
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim


def main():
    tokens = []
    load_dotenv()

    model = Word2Vec.load(r"C:\Users\Atman S\Documents\GitHub\Atlas\Atlas\atlas.model")
    
    # print(tokens)
    # model.build_vocab(tokens, progress_per=10)

    # model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)
    # print("Done")   

    # model.save("atlas.model")
    print(model.wv.most_similar("score"))
    

    
    

    

    # ask question
    # user_question = st.text_input("Ask a question about your PDF:")
    
    




        
        
        
        

    

if __name__ == '__main__':
    main()
    