import jerkbot_irc as jb
import openai

# Set up the OpenAI API client
openai.api_key = "API_KEY"

# create the bot and connect to the server
personality = "a sarcastic jerk"
channel = "#CHANNEL"
nickname = "NICKNAME"
#password = "PASSWORD" #comment out if unregistered
server = "SERVER"

#check if there is a password
try:
    bot = jb.AIBot(personality, channel, nickname, server, password)
except:
    bot = jb.AIBot(personality, channel, nickname, server)
    
bot.start()
