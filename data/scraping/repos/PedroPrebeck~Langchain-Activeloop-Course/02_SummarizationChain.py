from dotenv import load_dotenv

load_dotenv()

from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

llm = OpenAI(model="text-davinci-003", temperature=0.7)

summarize_chain = load_summarize_chain(llm)

pdf_path = "02_Example_Resume.pdf"

document_loader = PyPDFLoader(file_path=pdf_path)
document = document_loader.load()

summary = summarize_chain(document)
print(summary["output_text"])
