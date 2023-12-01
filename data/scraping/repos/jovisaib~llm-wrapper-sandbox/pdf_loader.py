from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate



loader = PyPDFLoader("invoice.pdf")
pages = loader.load()

# print(len(pages))
# print(pages)
# print(pages[0].page_content)


llm = OpenAI(temperature=.7)
template = """Extract invoice number, organization name, address, email, subtotal amount, balance due in {pages}
Output : entity : type
"""

prompt_template = PromptTemplate(input_variables=["pages"], template=template)
ner_chain = LLMChain(llm=llm, prompt=prompt_template)
result = ner_chain.run(pages=pages[0].page_content)
print(result)