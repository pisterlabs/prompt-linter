#!pip install langchain

from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import TextLoader
from PyPDF2 import PdfReader
api_key="hf_XXXXXXX" 
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
def get_llm_response(question, answer):
    template="Question: {question}\n{answer}"
    prompt = PromptTemplate(template=template, input_variables=["question", "answer"])
    llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":4000}, huggingfacehub_api_token=api_key))
    response = llm_chain.run(question=question, answer=answer)
    return response
reader = PdfReader("harrypotter.pdf") 
pdf_text = ""
print("Total pages=",len(reader.pages)) 
page_numbers_to_read = [0] 
for page in page_numbers_to_read:
    page_text = reader.pages[page].extract_text()
    print("Reading pages {} of :\n{}".format(page+1, str(len(reader.pages))))
    pdf_text += page_text
file1=open(r"pdf2text.txt","a")
file1.writelines(pdf_text)
print("file saved")
input_data=pdf_text
instruction="What are the main findings of this paper?"
response = get_llm_response(input_data, instruction)
print("LLM Response:\n",str(response))
