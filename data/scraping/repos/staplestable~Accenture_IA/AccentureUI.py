import openai
import json
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from pathlib import Path
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Lendo arquivo JSON
json_file = open('./api-keys/openai.json')
data = json.load(json_file)
json_file.close()

#Variáveis globais
texts, docs, =  '',[]

#Função para ler todos os arquivos PDFs no diretorio e alocar conteúdo na variável texts.
def read_files():
    texts = ''
    for pdf in Path('./pdfs').glob('**/*.pdf'):        
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
                texts += page.extract_text()
    return texts
  
#Função usando RecursiveCharacterTextSplitter do LangChain para dividir todo conteúdo da variável texts em chunks pequenos para depois fazer o embedding.      
def split_texts(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=390,
        chunk_overlap=3,   
        separators= ["\n"]
        )
    chunks = text_splitter.split_text(texts)
    return chunks

#Função para gerar a base de conhecimento(vector store) fazendo os embeddings dos chunks processados com  OpenAI e FAISS.
def get_vector_store(chunks):
    #Para fazermos o embeddings usamos o modelo da OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_apikey)
    #Usamos o módulo FAISS do LangChain para gerar um vector store relacionando os chunks e as embeddings
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

#Função que retorna um chain de perguntas e respostas com o tipo 'stuff' que serve para comentários pequenos retirados de outros textos.
def get_chain():
    #Estamos usando a LLM (large languange model) da OpenAI com a temperatura 0 que resulta em respostas mais conservadas a partir do texto.
    llm = OpenAI(temperature=0,openai_api_key=st.session_state.openai_apikey)
    
    #Chain é uma série de combinações de componentes/LLMs para fazer aplicações mais complexas.
    #O chain 'load_qa_chain ' com tipo 'Stuff' pega uma lista de documentos, insere eles em um prompt no modelo de pergunta e resposta e em seguida passa pro LLM.
    #https://python.langchain.com/docs/modules/chains/document/stuff
    chain = load_qa_chain(llm, chain_type='stuff')
    
    return chain

def main():
   st.set_page_config(page_title="FCI", page_icon=":computer:")
   st.title("Assistente Virtual - FCI :computer:")
   
   if "openai_apikey" not in st.session_state:
       if len(data['openai'][0]['api_key']) < 16:
           print("Especifique seu API_KEY no arquivo JSON da pasta API_KEYS")
           st.warning("Especifique seu API_KEY no arquivo JSON da pasta API_KEYS")
           st.stop()
           
       st.session_state.openai_apikey = data['openai'][0]['api_key']
       
   openai.api_key = st.session_state.openai_apikey

   if "knowledge_base" not in st.session_state:
       global texts, docs
       with st.spinner("Lendo os arquivos PDFs como fonte de conhecimento..."):
           texts = read_files()
       with st.spinner("Separando texto em chunks..."):
           chunks = split_texts(texts)
       with st.spinner("Fazendo a relação dos embeddings com os chunks..."):
           st.session_state.knowledge_base = get_vector_store(chunks)
       
   if "chain" not in st.session_state:
       with st.spinner("Definindo o método de chain..."):
           st.session_state.chain = get_chain()
       
   #Inicializa o histórico do chat.
   if "messages" not in st.session_state:
       st.session_state.messages = []
   
   #Para cada mensagem no histórico ele renderiza a mensagem e seu autor.
   for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])
 
   if prompt := st.chat_input("Pergunte algo relacionado a FCI"): 
       with st.chat_message("user"):
           st.markdown(prompt)
       st.session_state.messages.append({"role": "user", "content": prompt})
       
       if(len(prompt) <= 3):
           response = "Sua entrada precisa conter mais que 3 caracteres."
       else:
           docs = st.session_state.knowledge_base.similarity_search(prompt) #retorna os documentos que tenham similaridades do vector_store com a query do usuario
           response = st.session_state.chain.run(input_documents=docs, question=prompt)
           
       with st.chat_message("assistant"):
           st.markdown(response)
       st.session_state.messages.append({"role": "assistant", "content": response})
          

if __name__ == '__main__':
    main()