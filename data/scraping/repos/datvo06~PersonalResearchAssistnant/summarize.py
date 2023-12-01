from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
import os


# Set OPENAI_API_KEY
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
text_splitter = CharacterTextSplitter()

# Turn document into ascii character
with open("sample.txt", "r") as f:
    text = f.read().encode('utf-8', errors='ignore').decode('utf-8')
with open('sample_out.txt', 'w') as f:
    f.write(text)
    f.close()


text = text_splitter.split(text)
print("Text split into {} pieces".format(len(text)))
docs = [Document(page_content=text[i]) for i in range(len(text))]
chain = load_summarize_chain(llm, chain_type="map_reduce")
with get_openai_callback() as cb:
    result = chain.run(docs)
    print(result)
    print("tokens used", cb.total_tokens)
