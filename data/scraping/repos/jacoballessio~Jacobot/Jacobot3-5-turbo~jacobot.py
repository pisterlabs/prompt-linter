import openai
import queue
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import asyncio
import logging
from condense_chat import condense_chat


TOKEN = open("../telegram_token.txt", "r").read()

openai.api_key = open("../openai_key.txt", "r").read()
KNOWLEDGE = str("I am a chatbot named Jacobot. ")
async def start(update, context):
    await update.message.reply_text("Hello! I am Jacobot, how can I assist you today?")

async def respond_to_message(update, context):
    KNOWLEDGE = str("I am a chatbot named Jacobot. ")
    hist_nec = is_history_necessary(update.message.text)
    print("histroynec: "+str(hist_nec))
    if hist_nec == True:
        chat_id=str(update.message.chat_id)
        condensed_chat = condense_chat(chat_id)
        KNOWLEDGE = str("I am a chatbot named Jacobot. I also have this information on our chat history: \n"+condensed_chat)
        print("kn:"+KNOWLEDGE)
    print("kn:"+KNOWLEDGE)
    log_messages(update, context)
    respond = should_respond(update.message.text)
    print(respond)
    if respond == False:
        return
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Here is what I know: "+KNOWLEDGE+"\n\n new message from: "+update.message.from_user.first_name+":"+update.message.text+"\n Jacobot:",
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    ).get("choices")[0].text
    log_response(update, context, response)
    await update.message.reply_text(response)

def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    #all messages that mention jacobot or ask a question
    #filter for accepted chat ids
    application.add_handler(MessageHandler(filters=filters.TEXT & ~filters.COMMAND, callback=respond_to_message))
    application.run_polling()

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Define a function to handle the messages
def log_messages(update, context):
    # Get the message text
    message_text = update.message.text
    # Get the username of the sender
    username = update.message.from_user.first_name
    #get the chat id
    chat_id = update.message.chat_id
    #save to txt file
    time = update.message.date
    with open(str(chat_id)+'log.txt', 'a') as f:
        f.write(str(username)+str(time)+": "+message_text+"\n")
    # Log the message
    logger.info("Message from %s: %s", username, message_text)
def log_response(update, context, response):
    #get the chat id
    time = update.message.date
    chat_id = update.message.chat_id
    #save to txt file
    with open(str(chat_id)+'log.txt', 'a') as f:
        f.write("Jacobot: "+str(time)+response+"\n")

def should_respond(message):
    #ask gpt-3 if it should respond
    #get examples from should_respond_exp.txt

    examples = open("should_respond_exp.txt", "r").read()
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Each response by jacobot is a yes or no response to the question 'should I respond to the previous message?': "+examples + "\nJacob Allessio,\n" + message+"\n Jacobot, \n",
        max_tokens=500,
        n=1,
        stop='\n',
        temperature=0.5,
    ).get("choices")[0].text
    #if the response contains yes(upper or lower), return true
    print(response)
    #trim response to first word
    response = response.split()[0]
    print(response)
    #lowercase response
    response = response.lower()
    if "yes" in response:
        return True
    else:
        return False

def is_history_necessary(message):
    #ask gpt-3 if it needs to look at the history of the chat to give an accurate response
    #get examples from is_history_necessary_exp.txt
    examples = open("is_history_necessary_exp.txt", "r").read()
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Each response by jacobot is a yes or no response to the question 'should I look at the history of the chat to give an accurate response?' Here are some examples: "+examples + "\nJacob Allessio,\n" + message+"\n Jacobot, \n",
        max_tokens=500,
        n=1,
        stop='\n',
        temperature=0.5,
    ).get("choices")[0].text
    #if the response contains yes(upper or lower), return true
    print(response)
    #trim response to first word
    response = response.split()[0]
    print("hist"+response)
    #lowercase response
    response = response.lower()
    if "yes" in response:
        return True
    else:
        return False

if __name__ == "__main__":
    main()
