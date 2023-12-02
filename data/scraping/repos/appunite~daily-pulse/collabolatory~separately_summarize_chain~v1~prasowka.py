import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

template = """
Write two sentences funny summary of this document: {document}.
"""

beginning = "Write a catchy one sentence beginning of a company-wide message which will include a summary of the previous day's decision made. Don't mention any decisions that were made"
ending = "Write a catchy one sentence ending of a company-wide message which will include a summary of the previous day's decision made."

prompt = PromptTemplate(
    input_variables=["document"],
    template=template,
)

llm = OpenAI(temperature=0.7)
text_splitter = CharacterTextSplitter()
chain = LLMChain(llm=llm, prompt=prompt)
documents_folder = ""


def updates(summaries):
    return "\n".join(str(summary) for summary in summaries)


####
from langchain.docstore.document import Document

summaries = []
with os.scandir(documents_folder) as entries:
    for entry in entries:
        with open(f"{documents_folder}/{entry.name}") as f:
            adrs = f.read()
            texts = text_splitter.split_text(adrs)
            summary = chain.run(adrs)
            summaries.append(summary)

docs = [Document(page_content=t) for t in summaries]
prompt_template = """Write a two sentences summary separately for each of the following documents, write down each of them in the new line starting with sequential number:

{text}

\n
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
print(chain.run(docs))
