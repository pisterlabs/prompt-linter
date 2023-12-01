# -*- coding: utf-8 -*-

from langchain.document_loaders import TextLoader
loader = TextLoader('./demo.txt')

from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

query = 'what did the president say about Ketanji Brown Jackson'
index.query(query)

query = 'what did the president say about Ketanji Brown Jackson'
index.query_with_sources(query)

from langchain.chains.quesion_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type='stuff')
chain.run(input_documents=docs, question=query)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
chain = load_qa_with_sources_chain(llm, chain_type='stuff')
chain({'input_documents': docs, 'question': query}, return_only_outputs=True)

