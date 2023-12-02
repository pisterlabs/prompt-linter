from django.shortcuts import render

from django.http import JsonResponse

import os
from dotenv import load_dotenv

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate

# load document
loader = DirectoryLoader('support_guy/faq', glob='**/*.txt')
documents = loader.load()

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
You are a technical support engineer. As a polite, friendly, and slightly informal professional, you provide honest answers to customer inquiries. If you don't know the answer, you'll respond with "I don't know" rather than making up information.
Using this information, please respond to the following emails:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain = load_qa_chain(llm=OpenAI(openai_api_key = os.environ["OPENAI_API_KEY"], temperature=0.7, max_tokens=1000), chain_type="stuff", prompt=PROMPT)


def generate_response(request):
    email = request.GET.get('email', '')
    print(f"Received email: {email}")

    res = chain.run(input_documents=documents, question=(email))
    
    return JsonResponse({'response': res})