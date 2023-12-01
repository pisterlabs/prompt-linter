from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from .summary_prompt import prompt_result_conclusion,prompt_title_intro_method
import os
from dotenv import load_dotenv
import datetime
load_dotenv()


def remove_references(text):
    """Remove references from text."""
    references_index = str(text.upper()).find("REFERENCES")

    # If the "REFERENCES" section is found, remove the text after it
    if references_index != -1:
        text = text[:references_index]

    return text

def get_time_period():
    current_hour = datetime.datetime.now().hour

    if 0 <= current_hour < 12:
        return "Morning"
    elif 12 <= current_hour < 17:
        return "Afternoon"
    elif 17 <= current_hour <= 23:
        return "Evening"
    else:
        return "invalid time"


def summarizer(text,temparature = '0.5', custom_prompt = ''):
    return f"{text[:10]}"

class Chatbot:
    def __init__(self):
        self.knowledge_base = None

    def initialize_knowledge_base(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
        self.knowledge_base = FAISS.from_texts(chunks, embeddings)

    def summarize_prompter(self,prompt,temperature = '0.5'):
        if self.knowledge_base is None:
            return "Knowledge base not initialized. Please call 'initialize_knowledge_base' first."

        docs = self.knowledge_base.similarity_search(prompt)
        llm = OpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'),temperature=temperature)
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=prompt)

        return response
    
    def summarizer(self):

        final_summary = ''
        # title = str(self.summarize_prompter(prompt_title))
        intro_title_meth = str(self.summarize_prompter(prompt_title_intro_method))
        result_conclusion = str(self.summarize_prompter(prompt_result_conclusion))

        final_summary =intro_title_meth + '<br>' + result_conclusion

        return final_summary


    def chat(self, question,temperature = '0.5'):
        if self.knowledge_base is None:
            return "Knowledge base not initialized. Please call 'initialize_knowledge_base' first."

        docs = self.knowledge_base.similarity_search(question)
        llm = OpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'),temperature=temperature)
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=question)

        return response
    
    def custom_summarizer(self,prompt,temperature = '0.5',custom_prompt=''):
        if self.knowledge_base is None:
            return "Knowledge base not initialized. Please call 'initialize_knowledge_base' first."

        docs = self.knowledge_base.similarity_search(prompt)
        llm = OpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'),temperature=temperature)
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=custom_prompt)

        return response



