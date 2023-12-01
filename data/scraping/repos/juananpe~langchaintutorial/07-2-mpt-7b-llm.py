'''
pip install python-telegram-bot langchain faiss-cpu tiktoken
'''

import os
import logging
from dotenv import load_dotenv
from telegram  import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI, AI21
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

from ctransformers.langchain import CTransformers
from langchain.embeddings import HuggingFaceInstructEmbeddings


load_dotenv()

DATABASE = None

logging.basicConfig(
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")



async def load(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # loader = TextLoader('state_of_the_union.txt')

        url = "https://www.ehu.eus/documents/340468/2334257/Normativa_TFG_cas/d85cae6b-7940-47ed-9c08-c1585648efc4" # TFG Normativa
        loader = PyPDFLoader(url)
        
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)


        # instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
        #                                              model_kwargs={"device": "cpu"})

        global DATABASE
        DATABASE = FAISS.from_documents(docs, OpenAIEmbeddings())
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Document loaded!")



async def query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    llm = CTransformers(model='/tmp/mpt-7b-instruct.ggmlv3.q5_0.bin', 
                    model_type='mpt')
    
    chain = RetrievalQA.from_chain_type(llm=llm, 
                                  chain_type="stuff", 
                                  retriever=DATABASE.as_retriever())
    
    results = chain.run(update.message.text)
    text = results['output_text']
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


if __name__ == "__main__":
    application = ApplicationBuilder().token(
            os.getenv('TELEGRAM_BOT_TOKEN')).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('load', load))
    application.add_handler(CommandHandler('query', query))
    application.run_polling()


