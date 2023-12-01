import openai
import queue
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import asyncio
import logging
from condense_chat import condense_chat

TOKEN = open("telegram_token.txt", "r").read()

openai.api_key = open("openai_key.txt", "r").read()
KNOWLEDGE = str("I am a chatbot named Jacobot. ")
response_few_shot = "Jacob: I am going to the moon\nJacobot:[NO-RESPONSE]\nJacob: Hey matt, can you help me?\nJacobot: [NO-RESPONSE]"
async def start(update, context):
    await update.message.reply_text("Hello! I am Jacobot, how can I assist you today?")

#this variable holds the number of messages sent since the last time respond_to_message returned true

async def respond_to_message(update, context):
    
    chat_id=str(update.message.chat_id)
    condensed_chat = condense_chat(chat_id)
    KNOWLEDGE = str("I am a chatbot named Jacobot. I also have this information on our chat history: \n"+condensed_chat)
    print("kn:"+KNOWLEDGE)
    log_messages(update, context)
    respond = should_respond(update.message.text)
    print(respond)
    
    if respond == False:
        return
    knPrompt = "Here is what I know: "+KNOWLEDGE
    prompt=knPrompt+"\n\n I am going to send you a message that is part of a group chat. Your name is Jacobot, and you should decide whether or not to respond to each message, denoting a lack of response with '[NO-RESPONSE]' \nJacobot: Okay, send the message \nnew message from: "+update.message.from_user.first_name+":"+update.message.text+"\nJacobot: "
    print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        n=1,
        temperature=0,
    ).get("choices")[0].message.content
    log_response(update, context, response)
    if "[NO-RESPONSE]" not in response:
        await update.message.reply_text(response)
    else:
        print("no response", response)

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
num_messages_since_last_should_respond = 2
def should_respond(message):
    global num_messages_since_last_should_respond
    #if the message has "jacobot" or "?" in it or starts with "/"
    if "jacobot" in message.lower() or "???" in message or message.startswith("/") or "gpt" in message.lower():
        num_messages_since_last_should_respond = 0
        return True
    else:
        if num_messages_since_last_should_respond < 5:
            num_messages_since_last_should_respond += 1
            return True
        else:
            num_messages_since_last_should_respond += 1
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
