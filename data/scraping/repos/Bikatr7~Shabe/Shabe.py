## built-in modules
import requests
import json
import os
import base64
import time

## third-party modules
import openai
import backoff

from openai.error import APIConnectionError, APIError, AuthenticationError, ServiceUnavailableError, RateLimitError, Timeout


##-------------------start-of-Shabe---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Shabe:

    """
    
    Work in progress.

    """

##-------------------start-of-__init__()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self):

        with open('C:\\Users\\Tetra\\Desktop\\ShabeSetupUrl.txt', 'r') as file:
            url = file.read()

        with open('C:\\Users\\Tetra\\Desktop\\ShabeSetupToken.txt', 'r') as file:
            token = file.read()

        self.auth = {
            'authorization': token
        }

        self.url = url

        self.messages = []

        if(os.name == 'nt'):  ## Windows
            self.config_dir = os.path.join(os.environ['USERPROFILE'],"ShabeConfig")
        else:  ## Linux
            self.config_dir = os.path.join(os.path.expanduser("~"), "ShabeConfig")


##-------------------start-of-get_pass_twenty_messages()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def get_past_twenty_messages(self):
        messages = requests.get(self.url, headers=self.auth)
        result = json.loads(messages.text)

        messages_to_return_content = []
        messages_to_return_user = []

        for i, key in enumerate(result):
            messages_to_return_content.append(key['content'])
            messages_to_return_user.append(key['author']['username'])

            if i >= 19:  # Check if 5 messages have been collected
                break

        return messages_to_return_content, messages_to_return_user

    
##-------------------start-of-post_message()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def post_message(self, message):
        msg = {
            'content': message
        }

        requests.post(self.url, headers=self.auth, data=msg)

##-------------------start-of-initialize_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def initialize(self) -> None:

        """

        Sets the open api key.\n
        
        Parameters:\n
        self (object - Shabe) : the Shabe object.\n

        Returns:\n
        None\n

        """

        try:
            with open(os.path.join(self.config_dir,'GPTApiKey.txt'), 'r', encoding='utf-8') as file:  ## get saved api key if exists
                api_key = base64.b64decode((file.read()).encode('utf-8')).decode('utf-8')

            openai.api_key = api_key
        
            print("Used saved api key in " + os.path.join(self.config_dir,'GPTApiKey.txt')) ## if valid save the api key
            time.sleep(.7)

        except (FileNotFoundError,AuthenticationError): ## else try to get api key manually
                
            if(os.path.isfile("C:\\ProgramData\\Kudasai\\GPTApiKey.txt") == True): ## if the api key is in the old location, delete it
                os.remove("C:\\ProgramData\\Kudasai\\GPTApiKey.txt")
                print("r'C:\\ProgramData\\Kudasai\\GPTApiKey.txt' was deleted due to Kudasai switching to user storage\n")
                
            api_key = input("DO NOT DELETE YOUR COPY OF THE API KEY\n\nPlease enter the openapi key you have : ")

            try: ## if valid save the api key

                openai.api_key = api_key

                if(os.path.isdir(self.config_dir) == False):
                    os.mkdir(self.config_dir, 0o666)
                    print(self.config_dir + " created due to lack of the folder")

                    time.sleep(.1)
                            
                if(os.path.isfile(os.path.join(self.config_dir,'GPTApiKey.txt')) == False):
                    print(os.path.join(self.config_dir,'GPTApiKey.txt') + " was created due to lack of the file")

                    with open(os.path.join(self.config_dir,'GPTApiKey.txt'), 'w+', encoding='utf-8') as key: 
                        key.write(base64.b64encode(api_key.encode('utf-8')).decode('utf-8'))

                    time.sleep(.1)
                
            except AuthenticationError: ## if invalid key exit
                    
                os.system('cls')
                        
                print("Authorization error with creating openai, please double check your api key as it appears to be incorrect.\n")
                os.system('pause')
                        
                exit()

            except Exception as e: ## other error, alert user and raise it

                os.system('cls')
                        
                print("Unknown error with connecting to openai, The error is as follows " + str(e)  + "\nThe exception will now be raised.\n")
                os.system('pause')

                raise e
                    
##-------------------start-of-build_messages()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def build_messages(self, prompt) -> None:

        '''

        builds messages dict for ai\n
        
        Parameters:\n
        self (object - Kijiku) : the Kijiku object.\n

        Returns:\n
        None\n

        '''

        name = "Shabe"

        system_message = f"You are {name}, a discord user that it trying to talk as much as possible.\nYou are doing this through 'Userphone' a discord bot that lets you communicate across servers.\nYou will be provided with the past twenty messages that have been sent in the discord channel you are in, as well as the username of the person who sent them.Please note that Yggdrasil is the bot commanding userphone, so do not talk to it.\nType the '--userphone' command to start a new call or type the '--hangup' command to end your current call. Please type '--userphone' if you understand."

        system_msg = {}
        system_msg["role"] = "system"
        system_msg["content"] = system_message

        self.messages.append(system_msg)

        model_msg = {}
        model_msg["role"] = "user"
        model_msg["content"] = prompt

        self.messages.append(model_msg)

##-------------------start-of-translate_message()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ## backoff wrapper for retrying on errors
    @backoff.on_exception(backoff.expo, (ServiceUnavailableError, RateLimitError, Timeout, APIError, APIConnectionError))
    def get_gpt_response(self) -> str:

        '''

        translates system and user message\n

        Parameters:\n
        self (object - Kijiku) : the Kijiku object.\n
        system_message (dict) : the system message also known as the instructions\n
        user_message (dict) : the user message also known as the prompt\n

        Returns:\n
        output (string) a string that gpt gives to us also known as the translation\n

        '''

        ## max_tokens and logit bias are currently excluded due to a lack of need, and the fact that i am lazy

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=self.messages
            ,

        )

        ## note, pylance flags this as a 'GeneralTypeIssue', however i see nothing wrong with it, and it works fine
        output = response['choices'][0]['message']['content'] ## type: ignore
        
        return output

##-------------------start-of-main()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


client = Shabe()
client.initialize()

messages, usernames = client.get_past_twenty_messages()

##client.post_message("Dear god")

prompt = "\nPast twenty messages:\n"

for user, message in zip(usernames, messages):
    prompt += user + ": " + message + "\n"

client.build_messages(prompt)
print(client.get_gpt_response())
