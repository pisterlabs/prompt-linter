from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from llm.templates import prompt_template, condensed_prompt_template

def get_conversation_chain(vector_store):
  llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
  )

  QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
  CONDENSE_PROMPT = PromptTemplate(input_variables=["chat_history", "question"], template=condensed_prompt_template)

  memory = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key='question', 
    output_key='answer', 
    return_messages=True
  )

  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    verbose=False,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    condense_question_prompt = CONDENSE_PROMPT,
    memory=memory
  )

  return conversation_chain