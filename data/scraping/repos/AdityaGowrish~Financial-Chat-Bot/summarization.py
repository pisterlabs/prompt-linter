from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import llm_model

class Summary_Chain:
  def summary_chain(self,data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100,chunk_overlap = 30)
    texts = text_splitter.split_documents(data)
    llms =llm_model.Models()
    llm = llms.llm_model()
    chain = load_summarize_chain(llm, chain_type = "stuff", verbose = False)
    output_summary = chain.run(texts)
    print(output_summary)
