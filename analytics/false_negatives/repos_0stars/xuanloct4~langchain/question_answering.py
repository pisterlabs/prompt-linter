

import environment

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

with open("./documents/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

docsearch = Chroma.from_texts(texts, embedding, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()

query = "What did the president say about Justice Breyer"
docs = docsearch.get_relevant_documents(query)

from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="stuff")
query = "What did the president say about Justice Breyer"
chain.run(input_documents=docs, question=query)


chain = load_qa_chain(llm, chain_type="stuff")
query = "What did the president say about Justice Breyer"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Italian:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
chain({"input_documents": docs, "question": query}, return_only_outputs=True)


chain = load_qa_chain(llm, chain_type="map_reduce")
query = "What did the president say about Justice Breyer"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
chain = load_qa_chain(llm, chain_type="map_reduce", return_map_steps=True)
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text translated into italian.
{context}
Question: {question}
Relevant text, if any, in Italian:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer italian. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}
=========
{summaries}
=========
Answer in Italian:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)
chain = load_qa_chain(llm, chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
chain({"input_documents": docs, "question": query}, return_only_outputs=True)

# from langchain.llms import OpenAI
# llm = OpenAI(batch_size=5, temperature=0)



chain = load_qa_chain(llm, chain_type="refine")
query = "What did the president say about Justice Breyer"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
chain = load_qa_chain(llm, chain_type="refine", return_refine_steps=True)
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
refine_prompt_template = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer. Reply in Italian."
)
refine_prompt = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=refine_prompt_template,
)


initial_qa_template = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\nYour answer should be in Italian.\n"
)
initial_qa_prompt = PromptTemplate(
    input_variables=["context_str", "question"], template=initial_qa_template
)
chain = load_qa_chain(llm, chain_type="refine", return_refine_steps=True,
                     question_prompt=initial_qa_prompt, refine_prompt=refine_prompt)
chain({"input_documents": docs, "question": query}, return_only_outputs=True)



chain = load_qa_chain(llm, chain_type="map_rerank", return_intermediate_steps=True)
query = "What did the president say about Justice Breyer"
results = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
print(results["output_text"])
print(results["intermediate_steps"])
from langchain.output_parsers import RegexParser

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer In Italian: [answer here]
Score: [score between 0 and 100]

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer In Italian:"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=output_parser,
)

chain = load_qa_chain(llm, chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)
query = "What did the president say about Justice Breyer"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
