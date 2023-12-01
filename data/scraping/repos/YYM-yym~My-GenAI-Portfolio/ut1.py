# my_qa.py

import openai
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain import PromptTemplate

class yumoyaoqa:
    def __init__(self, file_path='cv.json'):
        # Initialize LLM
        self.llm = OpenAI(
            model="text-davinci-003",
            temperature=0,
            max_tokens=1024,
            openai_api_key="your-api-key"
        )

        # Load and split text from JSON
        with open(file_path, 'r') as f:
            my_text = json.load(f)
        my_text_str = json.dumps(my_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        texts = text_splitter.create_documents([my_text_str])

        # Embed the text
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)

        # Set up the prompt template
        self.prompt_template = '''\
            You are Yumo Yao (she/her/hers)'s professional assistant. Yumo is an aspiring AI engineer.\
            Answer the question based on her career aspired in the future - 
            As a Generative AI Engineer, she'll join the expansive Data Science, Artificial Intelligence, and Big Data (DAB) division. 
            she will thrive in a stimulating, innovative, and continuous learning atmosphere, while playing a pivotal role in shaping the future Intelligent Enterprise driven by AI.
            Answer based on her information in a smooth and interesting way. Do not just list out information only, tell a story. \
            Make sure that correct information is included in the answer. If you need to, you can ask for more information. \

            {context}

            Question: {question} 

            If you do not know or need more information, just act as a professional assistant. Give the result only.\
            Read carefully.\

        '''
        
        self.PROMPT = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": self.PROMPT}
        self.qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)
        
        # self.context = ''

    def ask(self, query):
        openai.api_key = "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"
        answer = self.qa.run(query)
        refine_prompt = f'''
            Refine the answer:
            - Answer the question as you're Yumo's professional assistant
            - change the tone into professional, friendly, and polite
            - make sure the answer is smooth and interesting
            - if the answer is longer than 3 sentences, summarize it
            - Give the result only. Do not start with "Answer:" or "Response:" or "Result:"

            {answer}
        '''

        message=[{"role": "user", "content": refine_prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = message,
            temperature=0.2,
            max_tokens=1000,
            frequency_penalty=0.0
        )

        response = response["choices"][0]["message"]["content"]
        # print(response)
        self.prompt_template += "\n" + response
        return response
