'''
Jerkbot, an OpenAI chatbot for the Matrix chat protocol.

Uses gpt-3.5-turbo to generate responses with a preset personality which can be changed.
Named for my chosen personality, sarcastic jerk.  You can set any personality type, character, inanimate object, place, concept, etc.
There is a list of random examples included in the prompts.py file which also includes some altered prompts from awesome-chatgpt-prompts

Written by vagabond @realvagabond:matrix.org
h1ddenpr0cess2085@gmail.com
@vvagabondd
March 2023

still has a few bugs to work on, mostly error handling stuff.
'''

from matrix_client.client import MatrixClient
from matrix_client.api import MatrixHttpApi
from matrix_client.user import User
import openai
import time
import datetime
import random
import prompts
import threading
import sys

class AIbot:
    def __init__(self, server, token, username, password, channel, personality):
        self.server = server
        self.username = username
        self.password = password
        self.channel = channel
        self.personality = personality
        self.token = token

        self.bot_id = "@" + self.username + ":" + self.server.lstrip("https://") #creates the bot's full username @username:server.com
                
        #matrix_client stuff
        self.client = MatrixClient(self.server)
        self.matrix = MatrixHttpApi(self.server, token=self.token)
        self.user = User(self.matrix, self.bot_id)

        #self.user.set_display_name(self.username)  #set to default display name
        
        #get the bot's display name
        self.display_name = self.user.get_friendly_name() 
        
        self.logged_in = False

        self.room_id = self.matrix.get_room_id(self.channel) #room bot is in
        self.join_time = datetime.datetime.now() #time bot joined

        self.messages = {} #keeps track of history

    def login(self):
        try:
            self.client.login(username=self.username, password=self.password)
            self.logged_in = True #login success
            self.room = self.client.join_room(self.channel) #Joins the channel
        except Exception as e:
            print(e)
            self.logged_in = False
            sys.exit()
            
    def get_display_names(self):
        members = self.matrix.get_room_members(self.room_id)
        self.display_names = {}
        for member in members["chunk"]:
            try: #if the username has a display name add them to the display_names list
                self.display_names.update({member["content"]["displayname"]:member["sender"]})
            except:
                pass
        
    #start bot
    def start(self):
        #Login to Matrix
        self.login()
        #Get the room members
        self.get_display_names()
        
        #Listen for incoming messages
        self.client.add_listener(self.handle_message)
        self.client.start_listener_thread()
        self.matrix.sync() #unsure if needed?
        
        self.matrix.send_message(self.room_id, "Hey, I'm {}, an OpenAI chatbot.  Type .help for more information.".format(self.display_name)) #optional entrance message

    #Stop bot   
    def stop(self):
        #add a check to see if the bot is the only user in the channel
        
        self.matrix.send_message(self.room_id, "Goodbye for now.") #optional exit message
        self.matrix.leave_room(self.room_id)
        

    #Sets the personality for the bot
    def persona(self, sender, persona):
        try:
            self.messages[sender].clear()
        except:
            pass
        #Custom prompt written by myself that seems to work nearly flawlessly if used correctly.
        personality = "assume the personality of " + persona + ".  roleplay and always stay in character unless instructed otherwise.  keep your first response short."
        self.add_history("system", sender, personality) #add to the message history
        
    
    def add_history(self, role, sender, message):
        if sender in self.messages: #if this user exists in the history dictionary
            self.messages[sender].append({"role": role, "content": message}) #add the message
        
        else:
            if role == "system":
                self.messages[sender] = [{"role": role, "content": message}]
            else: #add personality to the new user entry
                self.messages[sender] = [
                    {"role": "system", "content": "assume the personality of " + self.personality + ".  roleplay and always stay in character unless instructed otherwise.  keep your first response short."},
                    {"role": role, "content": message}]

    #Create AI response
    def respond(self, sender, message, sender2=None):
        try:
            #Generate response with gpt-3.5-turbo model, you can change to gpt-4 if you have access and want to spend the money.  i have access but i can't afford it.
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=message)
        except Exception as e:
            self.matrix.send_message(self.room_id, e)
        else:
            #Extract response text and add it to history
            response_text = response['choices'][0]['message']['content']
            self.add_history("assistant", sender, response_text)
            if sender2: #if the .x function was used
                display_name = self.matrix.get_display_name(sender2)
            else: #normal .ai response
                display_name = self.matrix.get_display_name(sender)
            response_text = display_name + ":\n" + response_text.strip()
            #Send response to channel
            try:
                self.matrix.send_message(self.room_id, response_text)
            except Exception as e: #fix?
                self.matrix.send_message(self.room_id, e)
            #Shrink history list
            if len(self.messages[sender]) > 16: #if this gets too big, you'll get a token error
                del self.messages[sender][1:3]  #delete the first set of question and answers 
        
    #OpenAI moderation endpoint, checks if it violates ToS
    def moderate(self, message):
        flagged = False
        if not flagged:
            moderate = openai.Moderation.create(input=message,) #run through the moderation endpoint
            flagged = moderate["results"][0]["flagged"] #true or false
        return flagged
    
    #Handles chat messages
    def handle_message(self, event):
        
        self.matrix.sync()
        
        #convert time to same format as self.join_time
        message_time = event["origin_server_ts"] / 1000
        message_time = datetime.datetime.fromtimestamp(message_time)
        
        if message_time > self.join_time: #if the message was sent after the bot joined the channel
                         
            sender = event["sender"] #format @username:matrix.org
            display_name = self.matrix.get_display_name(sender) #current display name for the sender
            
            #if event happened in specified room, and is a message that hasn't been removed (caused error), and sender isn't the bot
            if event["room_id"] == self.room_id and event["type"] == "m.room.message" and sender != self.bot_id:
                self.get_display_names() #update display names in case someone joined
                message = event["content"]["body"] #message text from matrix event
                
                try:
                    #Trigger bot response with .ai
                    if message.startswith(".ai ") or message.startswith(self.display_name + ": "):
                        if message.startswith(".ai "):
                            #Strips .ai from the message
                            message = message.lstrip(".ai")
                            message = message.strip()
                        else:
                            message = message.lstrip(self.display_name + ":")
                            message = message.strip()
                        flagged = self.moderate(message) #check with openai moderation endpoint
                        if flagged: #Flagged by moderation
                            self.matrix.send_message(self.room_id, "This message violates the OpenAI usage policy and was not sent.")
                        else:
                            #Adds the message to the history
                            self.add_history("user", sender, message)

                            #Start respond thread
                            thread = threading.Thread(target=self.respond, args=(sender,self.messages[sender]))
                            thread.start()
                            thread.join(timeout=30)

                    #Lets you use another user's history for one message for collaboration        
                    if message.startswith(".x "):
                        #Strips .x from the message
                        message = message.lstrip(".x")
                        message = message.strip() #needed because of some bug if i used a space after .x above
                        #check the name in the display name dictionary
                        for name in self.display_names:
                            if type(name) == str and message.startswith(name):
                                user = str(self.display_names[name])
                                message = message.lstrip(name) 
                        
                                flagged = self.moderate(message) #check with openai moderation endpoint
                                if flagged: #Flagged by moderation
                                    self.matrix.send_message(self.room_id, "This message violates the OpenAI usage policy and was not sent.")
                                else:
                                    if user in self.messages:
                                        #Adds the message to the history
                                        self.add_history("user", user, message)
                                        #start respond thread
                                        thread = threading.Thread(target=self.respond, args=(user, self.messages[user],), kwargs={'sender2': sender})
                                        thread.start()
                                        thread.join(timeout=30)
                                    else:
                                        pass
                            else:
                                pass
                        
                    #Resets bot back to default personality
                    elif message.startswith(".reset"):
                        if sender in self.messages:
                            self.messages[sender].clear()
                            self.persona(sender, self.personality) # Preset personality
                            
                        self.matrix.send_message(self.room_id, "Bot has been reset to {} for {}".format(self.personality, display_name))
                        

                    #Remove personality by clearing history
                    elif message.startswith(".stock"):
                        if sender in self.messages:                            
                            self.messages[sender].clear() #if they already exist, clear
                        else:
                            self.messages[sender] = [] #create the history entry for the user 
                            
                        self.matrix.send_message(self.room_id, "Stock settings applied for {}.".format(display_name)) 

                                        
                    #Set personality
                    elif message.startswith(".persona "):
                        message = message.lstrip(".persona")
                        message = message.strip()
                        flagged = self.moderate(message) #check with openai moderation endpoint
                        if flagged:
                            self.matrix.send_message(self.room_id, "This persona violates the OpenAI usage policy and has been rejected.  Choose a new persona.")
                        else:
                            self.persona(sender, message) #set personality
                            #start respond thread
                            thread = threading.Thread(target=self.respond, args=(sender,self.messages[sender]))
                            thread.start()
                            thread.join(timeout=30)
                            
                    elif message.startswith(".prompt "):
                        message = message.lstrip(".prompt")
                        message = message.strip() #needed for issue i had with previous line removing first letter of message
                        #matches a key in the prompts dictionary
                        if message in prompts.prompt.keys():
                            self.messages[sender].clear()
                            message = prompts.prompt[message] #select matching key from dictionary
                            
                            self.add_history("system", sender, message) #add prompt to history
                            #start respond thread
                            thread = threading.Thread(target=self.respond, args=(sender,self.messages[sender]))
                            thread.start()
                            thread.join(timeout=30)
                        #Prompts help lists the available commands
                        elif message == "help":
                            message = ""
                            for key in sorted(prompts.prompt.keys()):
                                message += (key + ", ") #create comma separate list of keys
                            self.matrix.send_message(self.room_id, message)
                    #Help menu
                    elif message.startswith(".help"):

                        #create list of keys in prompts
                        keylist = []
                        for key in prompts.prompt.keys():
                            keylist.append(key)
                            
                       #used below for .format
                        persona_ex1, persona_ex2, persona_ex3 = random.sample(prompts.persona_list, 3) #3 unique selections from persona examples
                        prompt_ex1, prompt_ex2, prompt_ex3 = random.sample(keylist, 3) #3 unique selections from available custom prompts

                        #Help text
                        self.matrix.send_message(self.room_id, '''{}, an OpenAI chatbot.

.ai <message> or {}: <message>
    Basic usage.
    Personality is preset by bot operator.
    This bot is {}.

.x <user> <message>
    This allows you to talk to another user's chat history.
    <user> is the display name of the user whose history you want to use
    
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
'''.format(self.display_name, self.display_name, self.personality, persona_ex1, persona_ex2, persona_ex3, prompt_ex1, prompt_ex2, prompt_ex3)) #enables dynamic examples that change each time you use the command
                     
                except Exception as e: #probably need to add specific exceptions, fix later
                    print(e)
                    

if __name__ == "__main__":
    
    
    # Initialize OpenAI
    openai.api_key = "API_KEY"

    #Set server, username, password, channel, default personality
    server = "https://matrix.org" #not tested with other servers
    token = "TOKEN" #Matrix access token found on settings page in element
    username = "USERNAME"
    password = "PASSWORD"
    channel = "#CHANNEL:SERVER.org"
    #Choose default personality.  Can be pretty much anything. Examples in prompts.py
    personality = "a sarcastic jerk"
    

    #create AIbot and start it
    bot = AIbot(server, token, username, password, channel, personality)

    bot.start()

        
    
    input("BOT RUNNING.  Press Enter to exit") #prevents the program from exiting
        
    #bot.stop()
    ''' If uncommented, the bot leaves the channel when the program exits.
    disabled due to a bug where if you create a channel with the account you use for this bot,
    and nobody else joins the channel, and then you press enter to exit the program,
    the channel will be empty and you can't join it again.
    
    Commented out, the bot's username will remain in the channel even if the bot is not running.
'''
