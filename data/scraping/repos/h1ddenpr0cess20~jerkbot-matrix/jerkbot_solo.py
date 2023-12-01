## Jerkbot, an OpenAI chatbot for the Matrix chat protocol.  Uses gpt-3.5-turbo to generate responses with a preset personality which can be changed.

from matrix_client.client import MatrixClient
from matrix_client.api import MatrixHttpApi
from matrix_client.user import User

import openai
import sys
import time
import datetime
import random
import prompts


class AIbot:
    def __init__(self, server, token, username, password, channel, personality):
        self.server = server
        self.username = username
        self.password = password
        self.channel = channel
        self.personality = personality
        self.token = token

        self.bot_id = "@" + self.username + ":" + self.server.lstrip("https://")
        
        self.client = MatrixClient(self.server)
        self.matrix = MatrixHttpApi(self.server, token=self.token)
        self.user = User(self.matrix, self.bot_id)
        
        self.logged_in = False

        self.display_name = self.user.get_display_name()

        self.room_id = self.matrix.get_room_id(self.channel) #room bot is in
        self.join_time = datetime.datetime.now() #time bot joined
        
        self.messages = [] #Keeps track of history
        self.persona(self.personality) #Set default personality

        
        
    def login(self):
        try:
            self.client.login(username=self.username, password=self.password)
            self.logged_in = True
            self.room = self.client.join_room(self.channel) #Joins the channel
        except Exception as e:
            print(e)
            self.logged_in = False
            sys.exit()

    #Sets the personality for the bot
    def persona(self, persona):
        self.messages.clear()
        #persona = persona
        personality = "assume the personality of " + persona + ".  roleplay and always stay in character unless instructed otherwise.  keep your first response short."
        self.messages.append({"role": "system", "content": personality})
        
    #OpenAI moderation endpoint
    def moderate(self, message):
        flagged = False
        if not flagged:
            moderate = openai.Moderation.create(input=message,) #run through the moderation endpoint
            flagged = moderate["results"][0]["flagged"] #true or false
        return flagged
    
    #Create AI response
    def respond(self, message):
        #Generate response with gpt-3.5-turbo model
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo", 
          messages=message)
        #Extract response text and add it to history
        response_text = response['choices'][0]['message']['content']
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text.strip()
    
    #Handles chat messages
    def handle_message(self, event):
        self.matrix.sync()
        

        #convert time to same format as self.join_time
        message_time = event["origin_server_ts"] / 1000
        message_time = datetime.datetime.fromtimestamp(message_time)
        
        if message_time > self.join_time: #if the message was sent after the bot joined the channel
        
            if event["type"] == "m.room.message" and event["sender"] != self.bot_id:
                message = event["content"]["body"]
                try:
                    
                    #Resets bot back to default personality
                    if message.startswith(".reset"):
                        self.messages.clear()
                        self.persona(self.personality)
                        self.matrix.send_message(self.room_id, "Reset to {}".format(self.personality))
                    #Remove personality by clearing history
                    elif message.startswith(".stock"):
                        self.messages.clear()
                        self.messages.append({"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."})
                        self.matrix.send_message(self.room_id, "Bot has been reset to stock ChatGPT settings")
                    #Set personality
                    elif message.startswith(".persona "):
                        message = message.lstrip(".persona")
                        flagged = self.moderate(message)
                        if flagged:
                            self.matrix.send_message(self.room_id, "This persona violates the OpenAI usage policy and has been rejected.  Choose a new persona.")
                        else:
                            self.persona(message)
                            response = self.respond(self.messages)
                            self.matrix.send_message(self.room_id, response)
                    elif message.startswith(".prompt "):
                        message = message.lstrip(".prompt")
                        message = message.strip() #needed for issue i had with previous line removing first letter of message
                        #matches a key in the prompts dictionary
                        if message in prompts.prompt.keys():
                            self.messages.clear()
                            message = prompts.prompt[message]
                            self.messages.append({"role": "system", "content": message})
                            response = self.respond(self.messages)
                            self.matrix.send_message(self.room_id, response)
                        elif message == "help":
                            message = ""
                            for key in sorted(prompts.prompt.keys()):
                                message += (key + ", ") #create comma separate list of keys
                            self.matrix.send_message(self.room_id, message)
                                
                            
                    #Help menu    
                    elif message.startswith(".help"):

                        keylist = []
                        for key in prompts.prompt.keys():
                            keylist.append(key)
                            
                       #used below for .format
                        persona_ex1, persona_ex2, persona_ex3 = random.sample(prompts.persona_list, 3) #3 unique selections from persona examples
                        prompt_ex1, prompt_ex2, prompt_ex3 = random.sample(keylist, 3) #3 unique selections from available custom prompts

                        self.matrix.send_message(self.room_id,
'''{}, an OpenAI chatbot.

Solo version, chat like normal and it responds.  This works best in a channel with just one person, or a few who are collaborating.

    Personality is preset by bot operator.
    This bot is {}.
   
.persona <personality type or character or inanimate object>
    Changes the personality.  It can be a character, personality type, object, idea.
    Don't use a custom prompt here.
    If you want to use a custom prompt, use .stock then use .ai <custom prompt>
    Examples:
        .persona {}
        .persona {}
        .persona {}
        
.reset
    Reset to preset personality
    
.stock
    Remove personality and reset to standard GPT settings
    
.prompt help
    Lists custom prompts available for functions not easily set with .persona.
    
.prompt <prompt>
    Use special prompt from list of prompts
    Examples:
        .prompt {}
        .prompt {}
        .prompt {}
'''.format(self.display_name, self.personality, persona_ex1, persona_ex2, persona_ex3, prompt_ex1, prompt_ex2, prompt_ex3))
                    else:
                         flagged = self.moderate(message) #check with openai moderation endpoint
                         if flagged: #Flagged by moderation
                            self.matrix.send_message(self.room_id, "This message violates the OpenAI usage policy and was not sent.")
                         else:
                            self.messages.append({"role": "user", "content": message})
                            response = self.respond(self.messages)
                            #Send response to channel
                            self.matrix.send_message(self.room_id, response)
                            #Shrink history list
                            if len(self.messages) >= 18:
                                del self.messages[1:3]
                       
                except Exception as e:
                    print(e)
                    sys.exit()
            
    def start(self):
        #Login to Matrix
        self.login()
        #Listen for incoming messages
        self.client.add_listener(self.handle_message)
        self.client.start_listener_thread()
        self.matrix.sync()

        self.matrix.send_message(self.room_id, "Hey, I'm {}, an OpenAI chatbot.  Type .help for more information.".format(self.display_name)) #optional entrance message

    #Stop bot   
    def stop(self):
        self.matrix.send_message(self.room_id, "Goodbye for now.") #optional exit message
        self.matrix.leave_room(self.room_id)
        
if __name__ == "__main__":
    # Initialize OpenAI
    openai.api_key = "API_KEY"

    #Set server, username, password, channel, default personality
    server = "https://matrix.org" #not tested with other servers
    token = "TOKEN" #Matrix access token found somewhere on settings page in element
    username = "USERNAME"
    password = "PASSWORD"
    channel = "#CHANNEL:SERVER.org"
    #Choose default personality.  Can be pretty much anything.  There are some examples in the help section above.
    personality = "a sarcastic jerk"
    
    #create AIbot and start it
    bot = AIbot(server, token, username, password, channel, personality)
    bot.start()


    input("BOT RUNNING.  press enter to quit") #prevents the program from exiting

    #bot.stop()
    ''' If uncommented, the bot leaves the channel when the program exits.
    disabled due to a bug where if you create a channel with the account you use for this bot,
    and nobody else joins the channel, and then you press enter to exit the program,
    the channel will be empty and you can't join it again.
    
    Commented out, the bot's username will remain in the channel even if the bot is not running.
'''

        
