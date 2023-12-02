from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History: 
{chat_history}
FollowUpInput: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about the Document you have uploaded. 
You are given the following extracted parts of a long document and a question. Provide a conversational answer. 
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.

Question: {question}
========
{context}
=========
Answer in Markdown: """
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm, 
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain