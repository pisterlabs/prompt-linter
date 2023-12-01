# Speech Recoginition libraries
import speech_recognition as sr
# from google.cloud import speech
import pyttsx3

# Chat Based Module
import openai

# Miscellaneous Libraries
import pandas as pd
import time
import os
# from speech_rec import main

# LangChain and SQLite 
import sqlite3
import pandas as pd
import os
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

# PYTTSX3 CLASS
class _TTS:
    '''
    Load the Engine separately to avoid the endless loop in runAndWait.
    '''
    engine = None
    rate = None
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', 'english+f6')
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', rate - 25)

    def say(self,text_):
        '''
        Speak up the input text.
        '''
        self.engine.say(text_)
        self.engine.runAndWait() # Wait till the engine stops executing.

# CHAT APP USING OPEN AI API
class ChatApp:
    '''
    Class which instantiates the openai API and handles the chat by providing a custom input
    received form voice based commands.
    '''
    def __init__(self):
        # Setting the API key to use the OpenAI API
        # openai.api_key = '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'
        #os.environ['OPENAI_API_KEY'] = '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'
        self.openai_key = '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'

        # Initializing the chatbot.
        self.messages = [
            {"role": "system", "content": "You are a dataframe wrangler to manipulate datasets."},
        ]
        self.flag = 0

        input_db = SQLDatabase.from_uri('sqlite:///auxel_db.sqlite3')

        llm_1 = OpenAI(openai_api_key=self.openai_key, temperature=0)

        self.db_agent = SQLDatabaseChain(llm=llm_1,
                                    database=input_db,
                                    verbose=True)
        
    def chat_davinci(self, message, df):
        openai_query = message

        df_main = df
        # Print the schema of the table
        schema = f"Schema of the table df_main:"
        for col in df_main.columns:
            # Check if column contains strings
            if df_main[col].dtype == object:
                # Check if column has less than 10 unique values
                if len(df_main[col].unique()) < 10:
                    # Print column name
                    schema +="\n{}: {}".format(col,", ".join(df_main[col].unique()))
                else:
                    schema += "\n{}".format(col)
            else:
                schema += "\n{}".format(col)

        # Use OpenAI's GPT-3 to generate SQL
        prompt = (f"Given the following database schema, write an SQL query :"
                f" {openai_query}\n\n"
                f"Database schema:\n\n{schema}\n\n"
                f"Select all the columns unless specified\n"
                f"This is my schema for search on a string use LIKE sql command rather than a query\n"
                f"Following are my intent classes\n"
                f"SHOW is displaying records/querying of a specific instance\n"
                f"SORT is sorting\n"
                f"OPERATION one which belongs of the other\n"
                f"FILTER is filtering of records\n"
                f"Produce the SQL Query and given the intent of {openai_query} in this format\n"
                f"Every query has one intent class\n"
                f"SQL Query|%%|Intent class:")
 
        # response = openai.Completion.create(
        #     engine="text-davinci-003",
        #     prompt=prompt,
        #     temperature=0.5,
        #     max_tokens=250,
        #     n=1,
        #     stop=None,
        # )

        # Connect sqlite database.
        self.conn = sqlite3.connect('auxel_db.sqlite3')

        # Print the generated SQL
        # open_ai_response = response.choices[0].text.strip()
        response = self.db_agent.run(openai_query)

        return response

    def chat(self, message):
        '''
        Call the chat endpoint from the API.
        '''

        self.messages.append({"role": "user", "content": message})

        # Get response from chat using the message list built so far.
        response = openai.Completion.create(
            model="text-davinci-003", #gpt-3.5-turbo
            messages=self.messages
        )
        
        print(response)
        # Append the response.
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]['content']
    
# AUXEL BOT
class Auxel:
    '''
    The driving class for the Auxel chatbot.
    '''
    init_flag = False
    def __init__(self):    
        # TODO: change to get dataset information from user.
        self.df = pd.read_csv('data.csv')
        # self.text = ''
        self.chat = ChatApp()
        # out = self.chat.chat(str(self.df)+'Remember the DataFrame.')
        # out = self.chat.chat_davinci(str(self.df)+'Remember the DataFrame.', self.df) # Davinci Version
        # print(out)

    def say(self, text):
        "Speak up the response and delete the instance formed."
        tts = _TTS()
        tts.say(text)
        del(tts)

    def listen(self):
        "Listen to user query."
        self.say('Heyy. How can I help you today?')
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio_data = r.listen(source,timeout=5)
            try:
                self.text = r.recognize_google(audio_data) # Use Free Google API for recognizing input audio data.
                self.text = self.process_input_query_1()
                return self.text
            
            except sr.UnknownValueError:
                print('Error: Speech recognition could not understand audio.')
            except sr.RequestError as e:
                print(f'Error: Could not request results from Speech Recognition service; {e}')

    # ============== TEST FUNCTION ===================================
    def process_input_query_1(self):
        out = self.chat.chat_davinci(self.text, self.df) # Davinci Version
        self.say(out)
        return out

    def process_input_query(self):
        "Process input query being converted to text using the Free Speech API from Google."
        if 'code' not in self.text:
            self.text += '. Just give me the output and do not give me the code.'

        if 'hello' in self.text or 'hey' in self.text or 'hi' in self.text:
            self.say('hello')
            return 'hello'

        if 'create' in self.text or 'table' in self.text:
             self.say('just a minute..')
             self.text += 'make the resultant dataframe comma seperated. Only give me the dataframe and no other text.'
             out = self.chat.chat(self.text)
             self.say('action performed!')
             return out

        # Not exiting the program.
        if 'bye' in self.text or 'byy' in self.text or 'by' in self.text or 'goodbye' in self.text:
            exit()

        print('Prompt: ',self.text)
        # out = self.chat.chat(self.text)
        out = self.chat.chat_davinci(self.text, self.df) # Davinci Version
        print('Output: ',out)
        
        if 'create' in self.text or 'prepare' in self.text:
            self.say('done!')
        
        else:
            self.say(out)
        
        return out
        
# def listen(bot):
#     # Separate function to listen using the Google Cloud Speech API.
#     bot.say('tell me what you want?')
#     transcript = main()
#     bot.text = transcript
#     out = bot.process_input_query()
#     return out