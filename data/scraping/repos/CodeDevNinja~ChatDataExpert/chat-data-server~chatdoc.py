from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQA
from chatgpt import  create_embddings, create_llm
import openai
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate

openai.log = "debug"

def qa_with_doc(documents, query):
        llm = create_llm()
        template = """"Given the following extracted parts of a long document and a question,response in chinese.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            QUESTION: {question}
            =========
            {summaries}
            =========
            FINAL ANSWER IN chinese:
            """
        PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
        chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=PROMPT, verbose=True)
        result = chain({"input_documents": documents, "question": query}, return_only_outputs=True)
        result = result.get('output_text').split('=========')[0]
        result = result.split('"""')[0]
        result = result.split("'''")[0]
        return result
