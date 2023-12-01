# this script generates questions out of the inserted documents. (currenty 102 question-answers)
# It is not using the pinecone index since there is no possibility to retrieve all documents from the pinecone database.
# The questions and answers are stored in a MongoDB database.
# this will be especially used to question the students as a chatbot and test their knowledge.

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from llama_index import SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.embeddings import OpenAIEmbeddings
from pymongo import MongoClient
import pinecone
from langchain.vectorstores import Pinecone
from langchain.evaluation import QAEvalChain, ContextQAEvalChain
from python_translator import Translator
from dotenv import load_dotenv
import os, random

def question_generator():
    load_dotenv()
    # connection to the mongodb database
    client = MongoClient(os.getenv('MONGO_CONNECTION_STRING'))

    # select database
    db = client["DBISquestions"]

    # select collection
    collection = db["questionAnswer"]
    
    # get all documents from the src/documents folder
    slides = SimpleDirectoryReader(os.getenv("SELECTED_FILES")).load_data()
    exercises = SimpleDirectoryReader(os.getenv("SELECTED_EXERCISES")).load_data()
    
    # retrieve texts from the documents
    text = ""
    for i in range(len(slides)):
        text += slides[i].text
    for i in range(len(exercises)):
        text += exercises[i].text
    
    # generate questions out of the text
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    qa = chain.run(text)
    
    # insert questions and answers into the database
    for i in range(len(qa)):
        collection.insert_one(qa[i])
    return 

def random_question_tool(input):
    client = MongoClient(os.getenv('MONGO_CONNECTION_STRING'))
    database_name = "DBISquestions"
    database_list = client.list_database_names()
    if database_name in database_list:
        db = client["DBISquestions"]
        col = db["questionAnswer"]
        x = col.count_documents({})
        questionanswers = col.find()
        a = random.randint(0, x)
        question = questionanswers[a].get("question")
        
        # translate the question into german
        translator = Translator()
        translation = str(translator.translate(question, "german", "english"))
        return translation
    else:
        question_generator()
        db = client["DBISquestions"]
        col = db["questionAnswer"]
        x = col.count_documents({})
        questionanswers = col.find()
        a = random.randint(0, x)
        question = questionanswers[a].get("question")
        
        # translate the question into german
        translator = Translator()
        translation = str(translator.translate(question, "german", "english"))
        return translation
        
def answer_comparison(input):
    question = input.get("question")
    answer = input.get("answer")
    
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    
    index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    # initialize embedding model
    embed = OpenAIEmbeddings(
        model = "text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    text_field = "text"
    
    # connect to index
    vector_store = Pinecone(index, embed.embed_query, text_field)
    
    retriever = vector_store.as_retriever()
    
    eval_chain = ContextQAEvalChain(llm=ChatOpenAI(temperature=0), retriever=retriever, chain_type="stuff")
    output = eval_chain.evaluate_strings(
                        input=question,
                        prediction=answer,
                        reference=retriever,
                    )
    
    return output
    