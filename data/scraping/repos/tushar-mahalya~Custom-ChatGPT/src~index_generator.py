from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter

def construct_index(data_path: str, api_key: str):
    
    with open(data_path, 'r') as file:
    	comments = file.read()
    
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name= "gpt-3.5-turbo",
                                                            chunk_size =  500,
                                                            chunk_overlap = 50,
                                                            allowed_special={"<|endoftext|>"})
    split_comments = text_splitter.split_text(comments)
    embeddings = OpenAIEmbeddings(openai_api_key = api_key,
                                  allowed_special={"<|endoftext|>"})
    comments_embedding = FAISS.from_texts(split_comments, embeddings)
    comments_embedding.save_local("faiss_index")
    print('\033[32m' + 'Text Embeddings created Successfully ! \nStored in \'faiss_index\' directory' + '\033[0m')
    
    return comments_embedding