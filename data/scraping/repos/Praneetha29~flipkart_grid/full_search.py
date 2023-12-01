from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from inputPage.ingest import load_documents, split_documents, load_document_batch, load_single_document

import os
from dotenv import load_dotenv, find_dotenv
import openai


from inputPage.constants import SOURCE_DIRECTORY

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

#

def main(query):
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    docs = text_splitter.split_documents(text_documents)
    print("docs have been created")


    question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to find the violation. 
        {context}
        rules: {question}
        Relevant text, if any:"""

    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )

    combine_prompt_template = """You are an helpful AI model that checks for user compliance, system privileges and rule violation in audit logs.You are given rules and context. Check if any rule is violated  in the context
IMPORTANT DO NOT ANSWER WITH "As an AI model..." anytime 
IMPORTANT when you find a violation, quote it and tell how it can be fixed 
Go line by line and check for violations. Make sure you do not miss a violation if there is one. 
Use the following context (delimited by <ctx></ctx>), rules (delimited by <rule></rule>) the chat history (delimited by <hs></hs>):
------
<rule>
{question}
</rule>
------
<ctx>
{summaries}
</ctx>
------
Violations:"""
        
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )


    # query = get_query()
    qa = load_qa_chain(OpenAI(temperature=1), chain_type="map_reduce", question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
    result = qa({"input_documents": docs, "question": query}, return_only_outputs=True)
    print(result)
    return (result)

# if __name__ == "__main__":
#     print("started")
#     main()
#     print("end")
