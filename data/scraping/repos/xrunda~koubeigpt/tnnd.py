
import os
import json
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator



os.environ["OPENAI_API_KEY"]='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'


with open("modelY_transformed_corrected.csv") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()





chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
query = "What did the president say about Justice Breyer"
docs = docsearch.get_relevant_documents(query)
chain.run(input_documents=docs, question=query)