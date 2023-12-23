
# %%
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from dbsetup import vectorstore


# %%
template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done.

Chat History:\"""
{chat_history}
\"""

Follow Up Input: \"""
{question}
\"""

Standalone question:"""

condense_question_prompt = PromptTemplate.from_template(template)

template = """You are a friendly, conversational retail shopping assistant. Use the following context including product names, descriptions, image and product URL's to show the shopper whats available, help find what they want, and answer any questions.
It's ok if you don't know the answer, also give reasons for recommending the product which you are about to suggest the customer. Always recommend one product and ask for more from the user. Always return the product URL of the single product you are recommending to the customers. Please don't include image URL in the response.

Context:\"""
{context}
\"""

Question:\"
\"""

Helpful Answer:"""

qa_prompt= PromptTemplate.from_template(template)


# define two LLM models from OpenAI
llm = OpenAI(temperature=0.3)

streaming_llm = OpenAI(
    streaming=True,
    verbose=True,
    temperature=0.3,
    max_tokens=1500
)

# use the LLM Chain to create a question creation chain
question_generator = LLMChain(
    llm=llm,
    prompt=condense_question_prompt
)

# use the streaming LLM to create a question answering chain
doc_chain = load_qa_chain(
    llm=streaming_llm,
    chain_type="stuff",
    prompt=qa_prompt
)


# %%
import json
from langchain.schema import BaseRetriever
from langchain.vectorstores import VectorStore
from langchain.schema import Document
from pydantic import BaseModel
 
class RedisProductRetriever(BaseRetriever, BaseModel):
    vectorstore: VectorStore
 
    class Config:
        arbitrary_types_allowed = True
 
    def combine_metadata(self, doc) -> str:
        metadata = doc.metadata
        return (
           "Product Name: " + metadata["product_name"] + ". " +
           "Product Description: " + metadata["description"] + ". " +
           "Product URL: " + metadata["product_url"] + "." +
           "image: " + metadata["image"] + "."
        )
 
    def get_relevant_documents(self, query):
        docs = []
        for doc in self.vectorstore.similarity_search(query):
            content = self.combine_metadata(doc)
            docs.append(Document(
                page_content=content,
                metadata=doc.metadata
            ))
 
        return docs

# %%
redis_product_retriever = RedisProductRetriever(vectorstore=vectorstore)
 
chatbot = ConversationalRetrievalChain(
    retriever=redis_product_retriever,
    combine_docs_chain=doc_chain,
    question_generator=question_generator
)

# %%
# create a chat history buffer
chat_history = []
# gather user input for the first question to kick off the bot
# question = input("Hi! What are you looking for today?")
# # keep the bot running in a loop to simulate a conversation
# while True:
#     result = chatbot(
#         {"question": question, "chat_history": chat_history}
#     )
#     print("user:", result["question"])
#     print("bot:", result["answer"])
#     print("\n")
#     chat_history.append((result["question"], result["answer"]))
#     question = input()

