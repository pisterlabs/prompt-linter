#Originally based on code from a blog post https://medium.com/python-in-plain-english/super-quick-fine-tuning-llama-2-0-on-cpu-with-personal-data-d2d284559f
#blog was from Ashhadul Islam
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# Define the path to the database
DB_FAISS_PATH = "vectorstores/db_faiss/"

# Define a custom prompt template to guide the bot's responses
custom_prompt_template = '''
Please carefully utilize the following details to provide a precise response to the user's query.

It is critical to provide information that is accurate. If the answer is not within the data presented, 
kindly acknowledge that the information is not available instead of speculating.

[Context]
Provided context: {context}

[Question]
User's query: {question}

Ensure to relay only the pertinent answer without any additional information.

[System Response]
'''

def set_custom_prompt():
    """
    This function creates a prompt template using the specified structure
    and returns it for later use in the question-answer chain.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    """
    This function loads a language model with specified parameters
    and returns the loaded model.
    """
    llm = CTransformers(
        model='llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        max_new_tokens=1024,
        temperature=0.1
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    """
    This function creates a question-answer chain using the given language model,
    prompt template, and database, returning the configured QA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def init_qa_bot():
    """
    This function initializes the question-answer bot by setting up the necessary 
    embeddings, database, language model, and QA chain, returning the prepared QA bot.
    """
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
    return qa_chain

# Initialize the bot once and reuse it
qa_bot = init_qa_bot()

def final_result(query):
    """
    This function accepts a query as input and uses the initialized QA bot
    to generate and return a response.
    """
    response = qa_bot({'query': query})
    return response 

@cl.on_chat_start
async def start():
    """
    This async function handles the start of a chat session, sending a welcome message 
    and setting up the initial state of the session.
    """
    chain = qa_bot
    msg = cl.Message(content="Firing up the PromptMule Competitive Analysis bot...")
    await msg.send()
    msg.content = "Hi, welcome to PromptMule Competitive Analysis bot. What should I think about?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    """
    This async function handles incoming messages, using the QA bot to generate responses
    and sending those responses back to the user.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += "\n\nReferences:\n"
        for i, source in enumerate(sources, 1):
            answer += f"\n[{i}] {source}"
    else:
        answer += "\n[No sources available]"


    await cl.Message(content=answer).send()
