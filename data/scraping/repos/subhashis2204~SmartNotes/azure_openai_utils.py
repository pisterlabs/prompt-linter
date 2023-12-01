import json
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
import openai
from langdetect import detect_langs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from transformers import GPT2TokenizerFast
from flask import jsonify

class FlashCardGenerator:
    def __init__(self, subscription_key, endpoint, deployment_name):
        self.llm = AzureChatOpenAI(
            openai_api_base=endpoint,
            openai_api_key=subscription_key,
            deployment_name=deployment_name,
            openai_api_type='azure',
            openai_api_version='2023-05-15'
        )
        self.embedder = OpenAIEmbeddings(
            deployment='openaiembedding2204', 
            openai_api_key=subscription_key, 
            openai_api_base=endpoint, 
            openai_api_type='azure', 
            openai_api_version='2023-05-15'
        )
        self.qa = None
        self.db = None
        self.chat_history = []

    def generate_flashcards(self):
        loader = TextLoader("output.txt", encoding='utf-8').load()
        answer = None

        print(loader)

        try:
            chain = load_qa_chain(llm=self.llm, chain_type="map_reduce")
            query = 'output : short questions and short answers in [{"question" : "question 1", "answer" : "answer to question 1"}, {...}] format'
            response = chain.run(input_documents=loader, question=query)

            print(response)
            answer = json.loads(response)

        except Exception as e:
            print(e)
            answer = []

        return answer
    
    def generate_summary(self):
        answer = None

        try:
            loader = TextLoader("output.txt")
            docs = loader.load_and_split()

            chain = load_summarize_chain(llm=self.llm, chain_type="map_reduce")
            summary = chain.run(docs)

            answer = summary
        except Exception as e:
            print(e)

            answer = 'Sorry, I am unable to summarize the given text'

        return answer
    
    def generate_vector_db(self):
        self.db = None
        text = ''
        with open('output.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        def count_tokens(text):
            return len(tokenizer.encode(text))
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=24,
            length_function=count_tokens
        )

        documents = text_splitter.create_documents([text])
    
        self.db = FAISS.from_documents(documents, self.embedder)

        self.qa = ConversationalRetrievalChain.from_llm(self.llm, self.db.as_retriever())

        return 
    
    def questions_and_answers(self, query):
        print(self.qa)

        result = self.qa({"question" : query, "chat_history" : self.chat_history})

        if result and result['answer']:
            return jsonify({"response" : result["answer"]}), 200
        else :
            return jsonify({"response" : "Sorry, I don't know the answer to that question"}), 500
        
        