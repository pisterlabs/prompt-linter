from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from llama2_loader import PIPELINE
from llama2_embedding import VECTORDB

LLAMA2 = HuggingFacePipeline(
    pipeline = PIPELINE,
    model_kwargs={'temperature':0}
)

PROMPT_TEMPLATE = """
    <s>[INST] <<SYS>>
    {{ You are a helpful AI Assistant, and make sure only facts are provided, and tells don't know when not able to answer based on privded input and history}}<<SYS>>
    ###

    Previous Conversation:
    '''
    {chat_history}
    '''

    {{{question}}}[/INST]

    """

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, 
    input_variables=['question', 'chat_history'])

CHAIN = ConversationalRetrievalChain.from_llm(
    llm=LLAMA2,
    retriever=VECTORDB.as_retriever(),
    condense_question_prompt=PROMPT,
    return_source_documents=True,
    return_generated_question=True)

def query(question, history):
    result = CHAIN({
        "question": question,
        "chat_history": history
        })
    return result