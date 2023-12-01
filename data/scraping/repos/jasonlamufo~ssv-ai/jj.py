import logging
import os
import time
import openai
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update, BotCommand, ChatAction
from telegram.ext import CommandHandler, Filters, MessageHandler, Updater, CallbackQueryHandler

TESTMODE = True
GOODBYE_MSG = "Thank you for using the travel planning bot. Have a great day! \n /start: back to the menu"

# Set up the loggers
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BotLogger')
logger.setLevel(logging.DEBUG)

# Add a StreamHandler to the loggers to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Add file handler for the logger
file_handler = logging.FileHandler('bot.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Configuration Constants
OPENAI_KEY = os.getenv("OPENAI_KEY") or input("Input OPENAI_KEY:")
TG_TOKEN = os.getenv("TG_TOKEN") or input("Input TG_TOKEN:")
MAX_TOKENS = int(os.getenv("MAX_TOKENS")) or int(input("Input MAX_TOKENS:"))
AI_MODEL = os.getenv("AI_MODEL") or input("Input AI_MODEL:")

SIMPLE_EXAMPLE_MESSAGE = [ {"role": "system", "content": "You are a helpful, pattern-following travel advisor. " +
    "Do not talk about anything other than travel. " +
    "Help people plan their perfect trip."},
    {"role": "system", "name": "example_user", "content": "Las Vegas, 1 day"},
    {"role": "system", "name": "example_assistant", "content": """
        Day 1:

        Morning:
        
        - Arrive and check into your hotel.
        - Visit the High Roller observation wheel.
        - Take a leisurely walk through the Bellagio Conservatory and Botanical Garden.
        
        Afternoon:
        
        - Enjoy a meal at a restaurant on the Las Vegas Strip.
        - Visit the Fremont Street Experience for the light show.
        - Catch a show at the Mirage volcano.
        - Experience an exciting zipline tour.
        - Explore the Neon Museum and learn about the history of Las Vegas.
        - Stroll through the Red Rock Canyon National Conservation Area.
        
        Evening:
        
        - Dine at a restaurant in Red Rock Canyon.
        - Return to your hotel to relax after your adventure-filled day."""},
                           ]
DETAILED_EXAMPLE_MESSAGE =[
    {"role": "system", "content": "You are a helpful, pattern-following travel advisor. " +
        "Do not talk about anything other than travel. " +
        "Help people plan their perfect trip."},
    {"role": "system", "name": "example_user", "content": "Las Vegas, 1 day"},
    {"role": "system", "name": "example_assistant", "content": """
    Day 1:
    08:00 AM
    Check in to your hotel
    Leave your luggage and head out to explore
    
    09:00 AM - 10:00 AM
    Visit the High Roller observation wheel
    Take the tram from the Linq Hotel & Casino to the High Roller
    Spend 30 minutes riding the High Roller and enjoying the views
    10:00 AM - 11:00 AM
    
    Walk through the Bellagio Conservatory and Botanical Garden
    Walk from the High Roller to the Bellagio Conservatory
    Spend 30 minutes walking through the gardens and admiring the flowers
    
    11:00 AM - 12:00 PM
    Have lunch at one of the many restaurants on the Las Vegas Strip
    Walk from the Bellagio Conservatory to the Fashion Show Mall
    Have lunch at one of the many restaurants in the Fashion Show Mall
    
    12:00 PM - 01:00 PM
    Visit the Fremont Street Experience
    Take the Deuce bus from the Fashion Show Mall to Fremont Street
    Spend 30 minutes walking through the Fremont Street Experience and watching the light show
    
    01:00 PM - 02:00 PM
    See a show at the Mirage volcano
    Walk from Fremont Street to the Mirage Hotel and Casino
    Watch the volcano erupt every 15 minutes
    
    02:00 PM - 03:00 PM
    Go on a zipline tour
    Take a taxi from the Mirage Hotel and Casino to the SlotZilla Zipline
    Spend 30 minutes ziplining over the Fremont Street Experience
    
    03:00 PM - 04:00 PM
    Visit the Neon Museum
    Take a taxi from the SlotZilla Zipline to the Neon Museum
    Spend 1 hour walking through the Neon Museum and learning about the history of Las Vegas
    04:00 PM - 05:00 PM
    
    Take a walk through Red Rock Canyon National Conservation Area
    Take a taxi from the Neon Museum to Red Rock Canyon National Conservation Area
    Spend 1 hour walking through the canyon and enjoying the scenery
    
    05:00 PM - 06:00 PM
    Have dinner at one of the many restaurants in Red Rock Canyon
    
    06:00 PM - 07:00 PM
    Head back to your hotel
    Take a taxi from Red Rock Canyon to your hotel
    This revised plan is much more realistic and enjoyable. You'll still be able to see a lot of Las Vegas, but you'll have plenty of time to relax and enjoy yourself as well."""}
]
openai.api_key = OPENAI_KEY

# Utility
def simple_prompt(message) -> list[str, int]:
    response = openai.ChatCompletion.create(      
        model=AI_MODEL,
        temperature=0.2,
        max_tokens=MAX_TOKENS,
        messages = SIMPLE_EXAMPLE_MESSAGE + [
            {"role": "user", "content": generate_prompt(message)}
        ]
    )
    return [response.choices[0].message.content, response["usage"]["prompt_tokens"]]

def detailed_prompt(message) -> list[str, int]:
    response = openai.ChatCompletion.create(      
        model=AI_MODEL,
        temperature=0.2,
        max_tokens=MAX_TOKENS,
        messages = DETAILED_EXAMPLE_MESSAGE + [
            {"role": "user", "content": generate_prompt(message)}
        ]
    )
    return [response.choices[0].message.content, response["usage"]["prompt_tokens"]]


def generate_prompt(message):
    return """Suggest a travel plan which includes detailed transportation between attractions and accomodation according to the following travel destination and days:{}
            Please make sure the plan is realistic and consider the transportation time is reasonable between attractions and accomodation.
            """.format(message)
     
def button(update: Update, context):
    query = update.callback_query
    query.answer()
    
    if query.data == "restart" and context.user_data.get("flow_state") != "destination":
        start_handler(update, context)

    elif query.data == "create_plan":
        context.user_data.clear()  # Clear user data for a new plan
        start_travel_plan(update, context)

    elif query.data == "exchange_rate":
        access_exchange_rate(update, context)

    elif context.user_data["flow_state"] == "schedule":
        if query.data == "detailed_schedule":
            # Send the "waiting" message with the response then prompt
            waiting_message = context.bot.send_message(chat_id=update.effective_chat.id, text="Processing your request...")
            
            # Send a "typing" action to indicate the bot is processing the request
            context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

            # Edit the "waiting" message with the actual response
            prompt = context.user_data["prompt"]
            response = detailed_prompt(prompt)
            context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=waiting_message.message_id, text=response[0])
            if TESTMODE: context.bot.send_message(chat_id=update.effective_chat.id, text="Prompt tokens counted by the OpenAI API: " + str(response[1]))
            context.bot.send_message(chat_id=update.effective_chat.id, text=GOODBYE_MSG)

        elif query.data == "end_chat":
            context.bot.send_message(chat_id=update.effective_chat.id, text=GOODBYE_MSG)

        del context.user_data["flow_state"]
        del context.user_data["prompt"]
        del context.user_data["destination"]
        del context.user_data["days"]
        del context.user_data["additional_info"]   
    
# Commands
commands = [
    BotCommand("start", "Start planning the trip"),
    BotCommand("help", "Get help"),
    BotCommand("create_plan", "Create a travel plan"),
    BotCommand("exchange_rate", "Access the latest exchange rates")
]

# Command handlers
def start_handler(update: Update, context):
    if context.args:
        command = context.args[0]
        if command == "create_plan":
            start_travel_plan(update, context)
        elif command == "exchange_rate":
            access_exchange_rate(update, context)
        else:
            help_handler(update, context)
    else:
        keyboard = [[InlineKeyboardButton("Create Traveling Plan", callback_data='create_plan')],
                    [InlineKeyboardButton("Access Latest Exchange Rates", callback_data='exchange_rate')]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        context.bot.send_message(chat_id=update.effective_chat.id, text="Please select an option:",
                                 reply_markup=reply_markup)
        logger.info(f'start command is called by {update.message.from_user.username}.')

def help_handler(update: Update, context):
    text = "Available Commands:\n"
    for command in commands:
        text += f"/{command.command} - {command.description}\n"
    context.bot.send_message(chat_id=update.effective_chat.id, text=text)
    logger.info(f'help command is called by {update.message.from_user.username}.')

def start_travel_plan(update: Update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Please enter your destination for traveling:")
    context.user_data["flow_state"] = "destination"

def handle_user_response(update: Update, context):
    if "flow_state" in context.user_data:
        flow_state = context.user_data["flow_state"]

        if flow_state == "destination":
            destination = update.message.text.strip()
            context.user_data["destination"] = destination

            context.bot.send_message(chat_id=update.effective_chat.id, text="Please enter the number of days for traveling:")
            context.user_data["flow_state"] = "days"

        elif flow_state == "days":
            days = update.message.text.strip()
            context.user_data["days"] = days
            context.bot.send_message(chat_id=update.effective_chat.id, text="Please provide any additional information (budget, arrival/departure time, purpose of travelling:)")
            context.user_data["flow_state"] = "additional_info"

        elif flow_state == "additional_info":
            additional_info = update.message.text.strip()
            context.user_data["additional_info"] = additional_info

            destination = context.user_data["destination"]
            days = context.user_data["days"]
            additional_info = context.user_data["additional_info"]

            # Generate response using OpenAI with the collected information
            prompt = f"Destination: {destination}\nDays: {days}\nAdditional Information: {additional_info}"
            context.user_data["prompt"] = prompt

            # Send the "waiting" message with the response then prompt
            waiting_message = context.bot.send_message(chat_id=update.effective_chat.id, text="Processing your request...")
            
            # Send a "typing" action to indicate the bot is processing the request
            context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

            # Edit the "waiting" message with the actual response
            response = simple_prompt(prompt)
            context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=waiting_message.message_id, text=response[0])
            if TESTMODE: context.bot.send_message(chat_id=update.effective_chat.id, text="Prompt tokens counted by the OpenAI API: " + str(response[1]))

            # Ask for detailed schedule
            keyboard = [[InlineKeyboardButton("Yes", callback_data='detailed_schedule')],
                        [InlineKeyboardButton("No", callback_data='end_chat')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.send_message(chat_id=update.effective_chat.id, text="Click 'Yes' if you need a detailed schedule.", reply_markup=reply_markup)

            context.user_data["flow_state"] = "schedule"
    
    else:
        print("ok")
        # Send the initial message
        initial_message = context.bot.send_message(chat_id=update.effective_chat.id, text="Initial message")
        print("sleep")
        time.sleep(1)
        # Update the message text
        context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=initial_message.message_id, text="Updated message")



    logger.info(f'Message received from {update.message.from_user.username}.')


def start_travel_plan(update: Update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Please enter your destination for traveling:")
    context.user_data["flow_state"] = "destination"

def access_exchange_rate(update: Update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Accessing the latest exchange rates...")
    # Implement the logic to access the exchange rates here
    # Send the response to the user
    context.bot.send_message(chat_id=update.effective_chat.id, text="Here are the latest exchange rates.")

        
def main():
    # Init the Bot
    bot = Bot(token=TG_TOKEN)

    # Set the list of supported commands
    bot.set_my_commands(commands)

    # Add the command handlers to the dispatcher
    updater = Updater(token=TG_TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start_handler))
    dispatcher.add_handler(CommandHandler("help", help_handler))
    dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), handle_user_response))
    dispatcher.add_handler(CallbackQueryHandler(button))

    # Start the bot
    updater.start_polling()
    logger.info('The bot is started.')

if __name__ == '__main__':
    main()
