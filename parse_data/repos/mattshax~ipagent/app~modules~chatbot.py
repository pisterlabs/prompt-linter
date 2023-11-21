import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import LLMChain
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain

class Chatbot:

    def __init__(self, model_name, temperature, vectors, context):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors
        self.context = context


    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        Chat History:
        {chat_history}
        Follow-up entry: {question}
        Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """You are a friendly conversational assistant named IPAgent, designed to answer questions and chat with the user from a contextual file.
        You receive data from a user's file and a question, you must help the user find the information they need. 
        Your answers must be user-friendly and respond to the user in the language they speak to you.
        Respond in the format with a summary of the results, then list relevant patents in bullet format with the patent_number and a short summary of the abstract. 
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        question: {question}
        =========
        context: {context}
        ======="""
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT,verbose=True)
        doc_chain = load_qa_chain(llm=llm, 
                                  prompt=self.QA_PROMPT,
                                  verbose=True,
                                  chain_type="stuff")

        chain = ConversationalRetrievalChain(
            retriever=retriever, combine_docs_chain=doc_chain, question_generator=question_generator, verbose=True, return_source_documents=True)

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)
        
        st.session_state["history"].append((query, result["answer"]))
        
        # count_tokens_chain(chain, chain_input)

        # append sources
        source_rows=list(set([str(doc.metadata['row']) for doc in result['source_documents']]))
        answer = result['answer']
        print('source_rows',source_rows)
        
        html_sources=[]
        for row in source_rows:
            i = int(row)
            print(i,self.context.iloc[i]['patent_number'])
            html_sources.append('<a href="https://patents.google.com/patent/%s/en" target="_blank">%s</a>' % (self.context.iloc[i]['patent_number'],self.context.iloc[i]['patent_number']))
        answer = answer + '\n\nSources:\n' + '\n '.join(html_sources)

        return answer


def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 

    
    
