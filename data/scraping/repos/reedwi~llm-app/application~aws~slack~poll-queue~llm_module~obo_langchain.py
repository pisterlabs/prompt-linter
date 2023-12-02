import pinecone
from langchain.chains import RetrievalQA, LLMChain, ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from datetime import datetime
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.docstore.document import Document



pinecone.init(api_key=os.getenv('PINECONE_API_KEY'),
              environment="us-east1-gcp")


def get_retriever(namespace):
    embeddings = OpenAIEmbeddings()
    vector_db = Pinecone.from_existing_index('obo-internal-slackbot', embedding=embeddings, namespace=namespace)
    return vector_db.as_retriever(search_kwargs={"k": 3}, qa_template=get_qa_template()['prompt'])

def get_qa_template(message=None):
    if message not in [None, '']:
        message = message
    else:
        message = 'I could not find any answers for your question'
    template = f"""You are a helpful assistant for a company that is having a conversation with a human. You are responsible for helping look through the documents you have available to try and find an answer. You also have access to the prior hsitory of the conversation.
    You know that the current datetime is {datetime.now()}.
    If you can't find an answer or document that matches, return a message saying "{message}". If you are asked to generate or think about things based on data you have access to, complete the task. 
    Return all answers in markdown format.

    {{context}}

    QUESTION: {{question}}
    FINAL ANSWER IN MARKDOWN FORMAT:"""

    PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
    chain_type_kwargs = {"prompt": PROMPT}
    return chain_type_kwargs

def get_qa_condense_template(message=None):
    if message not in [None, '']:
        message = message
    else:
        message = 'I could not find any answers for your question'
    CONDENSE_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    PROMPT = PromptTemplate(template=CONDENSE_PROMPT, input_variables=["question", "chat_history"])
    return PROMPT

def get_gpt_prompt():
    template = """
    You are a helpful assistant. Please try to answer all questions to the best of your ability.
    Please return all responses in markdown style formatting

    Chat History: {history}
    Question: {human_input}
    Assistant:"""
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )
    return prompt

def get_answer(namespace, question, chat_history, no_results=None):
    if namespace == 'chatgpt':
        memory = ConversationBufferMemory(memory_key="history")
    else:
        memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)
    
    if chat_history:
        len_ch = len(chat_history)
        for i, message in enumerate(chat_history):
            if i == len_ch - 1:
                if not message['bot']:
                    continue

            if i == 0:
                orginal_question = message['text'].split('*Question*: ')[1]
                memory.chat_memory.add_user_message(orginal_question)
            elif message['bot']:
                if '*Source(s)*' in message['text']:
                    answer = message['text'].split('*Source(s)*')[0]
                    memory.chat_memory.add_ai_message(answer)
                else:
                    memory.chat_memory.add_ai_message(message['text'])
            else:
                memory.chat_memory.add_user_message(message['text'])

    if namespace == 'chatgpt':
        chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0.1, model_name='gpt-4'), 
            prompt=get_gpt_prompt(), 
            memory=memory
        )
        with get_openai_callback() as cb:
            result = chatgpt_chain.predict(human_input=question)
            tokens_used = cb.total_tokens
        
        return {
            "answer": result,
            "response": result,
            "tokens_used": tokens_used,
            "chatgpt": True
        }
    elif namespace == 'summarize':
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(question)
        docs = [Document(page_content=t) for t in texts]
        summarize_chain = load_summarize_chain(
            llm=OpenAI(temperature=0.1),
            chain_type='map_reduce' 
        )
        with get_openai_callback() as cb:
            result = summarize_chain.run(docs)
            tokens_used = cb.total_tokens
        
        return {
            "answer": result,
            "response": result,
            "tokens_used": tokens_used,
            "chatgpt": True
        }

    else:
        retriever = get_retriever(namespace=namespace)
        chain_type_kwargs = get_qa_template()
        with get_openai_callback() as cb:
            # chain = RetrievalQA.from_chain_type(
            #     llm=OpenAI(temperature=0.1), 
            #     chain_type='stuff', 
            #     retriever=retriever,
            #     memory=memory,
            #     return_source_documents=True, 
            #     chain_type_kwargs=chain_type_kwargs
            # )
            chain = ConversationalRetrievalChain.from_llm(
                llm=OpenAI(temperature=0.1, model_name='gpt-4'),
                retriever=retriever,
                chain_type='stuff',
                memory=memory,
                verbose=False,
                return_source_documents=True,
                condense_question_prompt=get_qa_condense_template(message=no_results),
                combine_docs_chain_kwargs=get_qa_template(message=no_results)
            )
            # print(question)
            result = chain({'question': question})
            print(result)
            tokens_used = cb.total_tokens
        
        return {
            "answer": result['answer'],
            "response": result,
            "tokens_used": tokens_used,
            "chatgpt": False
        }
        
def get_personal_answer(namespace, question, chat_history):
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)
    
    if chat_history:
        len_ch = len(chat_history)
        for i, message in enumerate(chat_history):
            if i == 0:
                memory.chat_memory.add_user_message(message['text'])
            elif message['bot']:
                if '*Source(s)*' in message['text']:
                    answer = message['text'].split('*Source(s)*')[0]
                    memory.chat_memory.add_ai_message(answer)
                else:
                    memory.chat_memory.add_ai_message(message['text'])
            else:
                memory.chat_memory.add_user_message(message['text'])

    retriever = get_retriever(namespace=namespace)
    chain_type_kwargs = get_qa_template()
    with get_openai_callback() as cb:
        chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(temperature=0.1, model_name='gpt-4'),
            retriever=retriever,
            chain_type='stuff',
            memory=memory,
            verbose=False,
            return_source_documents=True,
            condense_question_prompt=get_qa_condense_template(),
            combine_docs_chain_kwargs=get_qa_template()
        )
        # print(question)
        result = chain({'question': question})
        print(result)
        tokens_used = cb.total_tokens
        
        return {
            "answer": result['answer'],
            "response": result,
            "tokens_used": tokens_used,
            "chatgpt": False
        }