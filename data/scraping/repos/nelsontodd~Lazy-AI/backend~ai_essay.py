import os
import openai
import constants
import utils
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import WikipediaAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import pathlib
import subprocess
import tempfile
from reportlab.pdfgen import canvas
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, ListFlowable, ListItem
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

utils.pdf_to_txt(utils.input_rel_path(constants.case_study_pdf), utils.output_rel_path(constants.case_study_pdf))
loader = TextLoader(utils.output_rel_path(constants.case_study_pdf, ".txt"))
sources = loader.load()#get_github_docs("yirenlu92", "deno-manual-forked")
source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
search_index = Chroma.from_documents(source_chunks, OpenAIEmbeddings(model="text-embedding-ada-002"))

essay_prompt = constants.essay_prompt
essay_intro_prompt = PromptTemplate(
    template=constants.ARGUMENTATIVE_ESSAY_INTRO_PARAGRAPH_YAML, input_variables=[ "prompt", "context"]
)

body_paragraph = PromptTemplate(
    template=constants.ARGUMENTATIVE_ESSAY_BODY_PARAGRAPH_YAML, input_variables=["order",       "thesis","context"]
)
conclusion_paragraph = PromptTemplate(template=constants.ARGUMENTATIVE_ESSAY_CONCLUSION_YAML,
        input_variables=["thesis", "context"])

llm = OpenAI(temperature=0, max_tokens=-1, )
intro_chain = LLMChain(llm=llm, prompt=essay_intro_prompt)
body_1_chain = LLMChain(llm=llm, prompt=body_paragraph)
body_2_chain = LLMChain(llm=llm, prompt=body_paragraph)
body_3_chain = LLMChain(llm=llm, prompt=body_paragraph)
conclusion_chain = LLMChain(llm=llm, prompt=conclusion_paragraph)

def gen_introduction(prompt):
    docs = search_index.similarity_search(prompt, k=1)
    inputs = [{"context": doc.page_content, "prompt": prompt} for doc in docs]
    return intro_chain.apply(inputs)

def gen_body(order, thesis, chain):
    docs = search_index.similarity_search(thesis, k=1)
    inputs = [{"order": order, "thesis": thesis, "context": doc.page_content} for doc in docs]
    return chain.apply(inputs)

def gen_conclusion(thesis):
    docs = search_index.similarity_search(thesis, k=1)
    inputs = [{"thesis": thesis, "context": doc.page_content} for doc in docs]
    return conclusion_chain.apply(inputs)

def gen_essay_txt(prompt):
    introduction = gen_introduction(prompt)[0]["text"]
    thesis = utils.promptGPT(constants.EXTRACT_TOPIC_SENTENCE, introduction)
    with open(constants.output_path+"essay.txt", 'w') as f:
        f.write(introduction)
        f.write("\n")
        f.write(gen_body("1", thesis)[0]["text"])
        f.write("\n")
        f.write(gen_body("2", thesis)[0]["text"])
        f.write("\n")
        f.write(gen_body("3", thesis)[0]["text"])
        f.write("\n")
        f.write(gen_conclusion(thesis)[0]["text"])

def revise_paragraph(paragraph):
    edit = utils.promptGPT("""You are a professional editor and a great essay writer. Edit
    this paragraph from an essay and make it better, and more professional. Be careful not to change the meaning or arguments.""", paragraph, model="gpt-4")
    return edit

if __name__ == '__main__':
    doc = utils.default_pdf_doc(utils.output_rel_path(constants.case_study_pdf+".pdf"))
    _items = [utils.pdf_title("Argumentative Essay"), Spacer(1, 24)]
    introduction = gen_introduction(essay_prompt)[0]["text"]
    _items.append(utils.pdf_paragraph(revise_paragraph(introduction)))
    _items.append(Spacer(1, 24))
    thesis = utils.promptGPT(constants.EXTRACT_TOPIC_SENTENCE, introduction)
    for order,chain in enumerate([body_1_chain, body_2_chain, body_3_chain]):
        body = revise_paragraph(gen_body(str(order+1), thesis, chain)[0]["text"])
        _items.append(utils.pdf_paragraph(body))
        _items.append(Spacer(1, 24))
    conclusion = revise_paragraph(gen_conclusion(thesis)[0]["text"])
    _items.append(utils.pdf_paragraph(conclusion))
    doc.build(_items)

    print('AI Argumentative Essay Generated for Document: {}'.format(constants.case_study_pdf))
