# Need to install the following packages:
# pip install langchain
# pip install openai
# pip install redis

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import os
from getpass import getpass
import openai
import redis
import json
import random
import threading

class Chatbot:
    def __init__(self, host='redis', port=6379):
        self.client = redis.StrictRedis(host=host, port=port)
        self.pubsub = self.client.pubsub()
        self.username = None
        self.populate_mock_weather_data()
        self.populate_mock_fun_facts()  
        self.shutdown_flag = False
        self.openai_chat = None
        

    def introduce(self):
        header = "="*30 + " Welcome to LoveYourself Chatbot! " + "="*30
        footer = "="*80
        intro = f"""
        {header}

        Hello! I'm Ricky, your virtual chat companion. 
        I'm here to assist you in various ways. Here's a quick guide on how to interact with me:

        General Commands:
        -----------------
        1. !help               - Displays a list of all available commands and their usage.
        2. !weather <city>     - Provides the current weather conditions for the specified city.
                                    Example: !weather NewYork
        3. !fact               - Delivers a random fun fact straight to your screen.
        4. !whoami             - Shows the information you've stored with me.

        Channel Commands:
        -----------------
        5. !join <channel>            - Join a specified channel to send and receive messages.
        6. !leave <channel>           - Leave a channel you previously joined.
        7. !send <channel> <message>  - Send a message to all users in a specified channel.
        8. !listen <channel>          - Start receiving messages from a particular channel.
        9. !pm <username> <message>   - Send a private message to a specific user.
        10. !who <channel>            - List all active users in a channel.

        Advanced Commands:
        ------------------
        11. !QA LLM <question>  - Asks a question to a language model and receives an answer. 
                                You will be prompted to enter your OpenAI API key securely.

        {footer}

        Now that you know the basics, let's get chatting!
        """
        print(intro)

    def identify(self, username):
        user_key = f"user:{username}"
        user_data = self.client.hgetall(user_key)
        if user_data:
            # Load existing user info
            self.username = username
            print(f"Welcome back, {username}!")
        else:
            # Ask for new user info
            age = input("Please enter your age: ")
            gender = input("Please enter your gender: ")
            location = input("Please enter your location: ")
            self.client.hset(user_key, mapping={
                "name": username,
                "age": age,
                "gender": gender,
                "location": location
            })
            self.username = username
            print(f"Welcome, {username}!")

    def populate_mock_weather_data(self):
        cities = [
            "Beijing", "Hong Kong", "Los Angeles", "London", 
            "New York", "Paris", "Tokyo", "Moscow", "Berlin", 
            "Istanbul", "New Delhi", "Sydney", "Melbourne", 
            "Rio de Janeiro", "Sao Paulo", "Cape Town", "Johannesburg", 
            "Dubai", "Toronto", "Vancouver", "Mexico City", 
            "Singapore", "Bangkok", "Seoul", "Jakarta", 
            "Rome", "Madrid", "Barcelona", "Mumbai", "Kolkata", 
            "Buenos Aires", "Lima", "Cairo", "Tel Aviv", "Jerusalem", 
            "Lagos", "Nairobi", "Hanoi", "Ho Chi Minh City", 
            "Manila", "Kuala Lumpur", "Santiago", "Bogota", "Caracas", 
            "Oslo", "Stockholm", "Helsinki", "Copenhagen", "Amsterdam", 
            "Brussels"
        ]

        for city in cities:
            weather_data = {
                'temperature': str(random.randint(-20, 40)) + "°C",
                'condition': random.choice(["Sunny", "Rainy", "Snowy", "Cloudy", "Windy"]),
            }
            self.client.hset(f"weather:{city}", mapping=weather_data)

    def populate_mock_fun_facts(self):
        fun_facts = [
        "The harmonica is the world's best-selling musical instrument.",
        "Beethoven was completely deaf by the time he composed some of his most famous works.",
        "Elvis Presley was naturally blonde and dyed his hair black.",
        "The world's largest piano was built by Adrian Mann and is 5.7 meters long.",
        "Jimi Hendrix couldn't read or write music.",
        "The French horn is the hardest instrument to play, according to many musicians.",
        "The 'F' in Fender stands for Leo Fender, who founded the company but couldn't actually play the guitar.",
        "The longest recorded piece of music is John Cage's 'As Slow As Possible,' which lasts for 639 years.",
        "The didgeridoo is considered one of the world's oldest instruments, dating back over 40,000 years.",
        "Listening to music can actually lower levels of the stress hormone, cortisol.",
        "The most expensive musical instrument ever sold is a Stradivarius violin for $16 million.",
        "A 'jiffy' in musical terms is an actual unit of time for 1/100th of a second.",
        "The Beatles used the word 'love' 613 times in their songs.",
        "The British Navy uses Britney Spears' songs to scare off Somali pirates.",
        "Mozart could write music before he could write words."
        ]
        self.client.delete("fun_facts")  # Delete existing list, if any
        self.client.rpush("fun_facts", *fun_facts)  # Populate the list in Redis

    def process_commands(self, message):
        if message.lower() == 'exit':
            self.shutdown_flag = True 
            self.direct_message("Goodbye!")
            return
        # Handle special chatbot commands
        if message.startswith("!help"):
            self.direct_message("Here's a list of commands: !help, !weather, !fact, !whoami, !join, !leave, !send, !listen, !pm, !QA LLM, exit")
        elif message.startswith("!weather"):
            city = message[len("!weather "):] 
            city = city.strip('\'"')  # Remove quotes
            weather_data = self.client.hgetall(f"weather:{city}")
            if weather_data:
                weather_info = {k.decode('utf-8'): v.decode('utf-8') for k, v in weather_data.items()}
                self.direct_message(f"Weather in {city}: {weather_info['temperature']}, {weather_info['condition']}")
            else:
                self.direct_message(f"Sorry, weather data for {city} is not available.")
        elif message.startswith("!fact"):
            random_index = random.randint(0, self.client.llen("fun_facts") - 1)
            fact = self.client.lindex("fun_facts", random_index).decode('utf-8')
            self.direct_message(fact)
        elif message.startswith("!whoami"):
            if self.username:
                user_data = self.client.hgetall(f"user:{self.username}")
                user_info = {k.decode('utf-8'): v.decode('utf-8') for k, v in user_data.items()}
                self.direct_message(f"Your information: {user_info}")
        elif message.startswith("!join"):
            channel = message.split(" ")[1]
            self.join_channel(channel)
            print(f"Joined channel {channel}")
        elif message.startswith("!leave"):
            channel = message.split(" ")[1]
            self.leave_channel(channel)
            print(f"Left channel {channel}")
        elif message.startswith("!send"):
            tokens = message.split(" ")
            if len(tokens) < 3:
                print("Usage: !send <channel> <message>")
            else:
                channel, msg = tokens[1], " ".join(tokens[2:])
                self.send_message(channel, msg)
        elif message.startswith("!pm"):
            tokens = message.split(" ")
            if len(tokens) < 3:
                print("Usage: !pm <username> <message>")
            else:
                to_user, msg = tokens[1], " ".join(tokens[2:])
                self.send_private_message(to_user, msg)
        elif message.startswith("!who"):
            channel = message.split(" ")[1]
            self.who_is_in_channel(channel)
        
        elif message.startswith("!QA LLM"):
            question = message[len("!QA LLM "):]
            question = question.strip('\'"')
            
            if self.openai_chat is None:
                openai_api_key = getpass("Please enter your OpenAI API key: ")
                self.initialize_openai_chat(openai_api_key)

            prompt_str = """
            You are an expert that knows many things about this world. Please answer the questions:
            {input}
            """
            prompt_str_template = ChatPromptTemplate.from_template(prompt_str)
            test_input = prompt_str_template.format_messages(input=question)
            response = self.openai_chat(test_input)
            self.direct_message(response.content)
    
    def join_channel(self, channel):
        # Join a channel and add user to channel's active users
        channels_key = f"channels:{self.username}"
        self.client.sadd(channels_key, channel)
        self.client.sadd(f"active_users:{channel}", self.username)

    def leave_channel(self, channel):
        # Leave a channel and remove user from channel's active users
        channels_key = f"channels:{self.username}"
        self.client.srem(channels_key, channel)
        self.client.srem(f"active_users:{channel}", self.username)

    def who_is_in_channel(self, channel):
        # Fetch and display active users in the channel
        active_users = self.client.smembers(f"active_users:{channel}")
        active_users = [user.decode('utf-8') for user in active_users]
        self.direct_message(f"Active users in {channel}: {', '.join(active_users)}")

    def send_message(self, channel, message):
        # Send a message to a channel
        message_obj = {
            "from": self.username,
            "message": message
        }
        self.client.publish(channel, json.dumps(message_obj))

    def read_message(self, channel):
        # Read messages from a channel
        print(f"Sending messages to channel: {channel} ...")
        while True:
            message = input("Enter your message: ")
            self.client.publish(channel, message)

    def send_private_message(self, to_user, message):
        message_obj = {
            "from": self.username,
            "message": message
        }
        self.client.publish(to_user, json.dumps(message_obj))

    def direct_message(self, message):
        print(f"\033[92mChatbot> {message}\033[0m") 
        message_obj = {
            "from": "Chatbot",
            "message": message
        }
        direct_message_channel = f"direct_message:{self.username}"
        self.client.publish(direct_message_channel, json.dumps(message_obj))
    
    def get_user_info(self, user):
        user_key = f"user:{user}"
        return self.client.hgetall(user_key)

    def listen_to_channel(self, channel):
        # Subscribe to the channel
        self.pubsub.subscribe(channel)

        def message_handler():
            try:
                for message in self.pubsub.listen():
                    if self.shutdown_flag:  # Check flag
                        break
                    if message['type'] == 'message':
                        try:
                            msg_data = json.loads(message['data'])
                            print(f"{msg_data['from']}> {msg_data['message']}")
                        except json.JSONDecodeError:
                            print("Received an improperly formatted message.")
            except Exception as e:
                print(f"An error occurred: {e}")

        t = threading.Thread(target=message_handler)
        t.start()
        
    def initialize_openai_chat(self, api_key):
        print(f"Received API key: {api_key}")  # Debugging
        self.openai_chat = ChatOpenAI(temperature=0.0, model_name='gpt-4', openai_api_key=api_key)


if __name__ == "__main__":
    bot = Chatbot()
    bot.introduce()
    
    # Identify user by username
    username = input("Please enter your username: ")
    bot.identify(username)

    # Main interaction loop
    while True:
        message = input(f"{username}> ")

        if message.lower() == "exit":
            bot.shutdown_flag = True  # Set the flag to shut down listening thread
            print("Exiting the chat. Goodbye!")
            break
        elif message.startswith('!'):
            bot.process_commands(message)
        else:
            print(f"You said: {message}")

        # Add a conditional to start listening to a channel
        if message.startswith("!listen"):
            channel = message.split(" ")[1]
            bot.listen_to_channel(channel)

        # Subscribe to private messages, assuming usernames are unique
        bot.listen_to_channel(username)

