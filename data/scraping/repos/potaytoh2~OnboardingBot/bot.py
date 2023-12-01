from dotenv import dotenv_values
import openai
import logging
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes , MessageHandler, filters, InlineQueryHandler
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from tabulate import tabulate
from google.cloud import translate_v2 as translate
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.langchain import RagasEvaluatorChain
import os


config = dotenv_values(".env")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

#Initialize OPENAI and config
openai.api_key = config["OPENAI_API_KEY"]
MODEL_ENGINE = "gpt-3.5-turbo"

count = 0

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key='question',
    output_key='answer'
)

embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
db3 = Chroma(persist_directory="./VectorStore",embedding_function=embedding)

template = """ You are a chatbot for HealthServe, a non-profit organization that provides medical care to migrant workers in Singapore. 
Your role is to provide information regarding onboarding for new volunteers.
Given the following context and chat history, come up with an answer at the end.
{context}
Avoid answering questions that have nothing to do with HealthServe so if you encounter such questions, tell them that you're unable to answer a question that is out of scope.
Question: {question}
Helpful Answer:"""

custom_prompt = PromptTemplate.from_template(template)

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0.5,openai_api_key=openai.api_key)

#Create google cloud client
translate_client = translate.Client()

#Make Evaluation Chain
eval_chains = {
    m.name: RagasEvaluatorChain(metric=m) 
    for m in [faithfulness, context_precision, context_recall]
}

#we define a function that should process a specific type of update:
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=
        """Hi there I'm HealthServeAI, ask me any questions about HealthServe. I'm here to ensure you have a proper onboarding!""")
    
async def getResponse(user_id:int ,user_input: str, source_language: str):
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    try:
        answer = query(user_input)
        # If the source language is not English, translate the response to the source language
        if source_language != 'en':
            answer = translate_to_source_language(answer, source_language)
        return answer
    except Exception as error_msg:
        return "Sorry I was unable to generate a response from openAI"

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): # Define the chat message handler
    messageFromUser = update.message.text.strip()
    user_id = update.message.from_user.id

    # Detect the language of the message
    source_language = detect_language(messageFromUser)

    # If message is not in English, translate it to English
    if source_language != 'en':
        messageFromUser = translate_to_english(messageFromUser, source_language)
    responseFromOpenAi = await getResponse(user_id,messageFromUser,source_language)
    await update.message.reply_text(responseFromOpenAi) # Send the response to the user

async def caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_caps = ' '.join(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

async def inline_caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    if not query:
        return
    results = []
    results.append(
        InlineQueryResultArticle(
            id=query.upper(),
            title='Caps',
            input_message_content=InputTextMessageContent(query.upper())
        )
    )
    await context.bot.answer_inline_query(update.inline_query.id, results)

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")

def query(question:str):
    chunk(question,4)
    retriever=db3.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k": 10})
    qa = ConversationalRetrievalChain.from_llm(
    llm,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True,
    memory=memory
    )
    result = qa({"question":question})
    evalChainRun(result)
    # ,"chat_history":chat_history
    return result["answer"]

def chunk(question:str, k:int):
    try:
        global count
        docs = db3.similarity_search(question, k)
        table_data = []

        for i in range(k):
            metadata = docs[i].metadata
            page_content = docs[i].page_content
            source = metadata['source']
            chunk_data = [i, source, page_content]
            table_data.append(chunk_data)

        table_str = tabulate(table_data, headers=["Index", "Source", "Page Content"], tablefmt="grid")
        
        with open(f"queries/${count}.html", "w", encoding='utf-8') as html_file:
            html_file.write(f"<html><body>{table_str}</body></html>")
        count += 1


    except Exception as error_msg:
        print("Error: ", error_msg)

def evalChainRun(result):
    try:
       fakeResult = result.copy()
       query_key = 'query'
       result_key = 'result'
       fakeResult[result_key] = fakeResult.pop('answer')
       fakeResult[query_key] = fakeResult.pop('question')
       for name, eval_chain in eval_chains.items():
            score_name = f"{name}_score"
            print(f"{score_name}: {eval_chain(fakeResult)[score_name]}")

    except Exception as e:
        print(e)


# for google cloud detection of language
def detect_language(text):
    # Initialize the Google Cloud Translation client
    client = translate.Client()

    # Detect the language of the text
    result = client.detect_language(text)
    
    return result['language']

# for google cloud translation to english
def translate_to_english(text, source_language):
    # Initialize the Google Cloud Translation client
    client = translate.Client()

    # If the source language is English, return the text as is
    if source_language == 'en':
        return text

    # Translate the text to English
    translation = client.translate(text, source_language=source_language, target_language='en')
    
    return translation['translatedText']

# for google cloud translation to source language
def translate_to_source_language(text, source_language):
    # Initialize the Google Cloud Translation client
    client = translate.Client()

    # If the source language is English, return the text as is
    if source_language == 'en':
        return text

    # Translate the text back to the source language
    translation = client.translate(text, source_language='en', target_language=source_language)
    
    return translation['translatedText']



if __name__ == '__main__':
    application = ApplicationBuilder().token(config["TELEGRAM_API_KEY"]).build()

   
    caps_handler = CommandHandler('caps', caps)
    chat_handler = MessageHandler(filters.TEXT &  (~filters.COMMAND), chat_handler)
    inline_caps_handler = InlineQueryHandler(inline_caps)
    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    start_handler = CommandHandler('start', start)
    

    application.add_handler(start_handler)
    application.add_handler(caps_handler)
    application.add_handler(inline_caps_handler)
    #This handler must be added last
    application.add_handler(chat_handler)
    application.add_handler(unknown_handler)

    application.run_polling()
