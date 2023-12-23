from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate

def get_conversation_chain(vector_store):
  llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
  )

  prompt_template = """You are a helpful AI assistant. 
  Use the following pieces of context to answer the question at the end. 
  If you don't know the answer, just say you don't know. DO NOT try to make up an answer. 
  Don't give information not mentioned in the CONTEXT INFORMATION.

  {context}

  Question: {question}
  Helpful answer in markdown:
  """
  QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

  condensed_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question: 
  """
  CONDENSE_PROMPT = PromptTemplate(input_variables=["chat_history", "question"], template=condensed_prompt_template)

  memory = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key='question', 
    output_key='answer', 
    return_messages=True
  )

  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    # chain_type="map_reduce",
    # verbose=True,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    condense_question_prompt = CONDENSE_PROMPT,
    memory=memory
  )
  # print(conversation_chain)
  return conversation_chain