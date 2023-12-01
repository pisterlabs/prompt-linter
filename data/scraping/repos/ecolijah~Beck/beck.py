import tkinter as tk
import speech_recognition as sr
import os
import openai
import threading
import pinecone
from time import time
from uuid import uuid4
from dotenv import load_dotenv
from class_file import RollingMessages, TextToSpeech
from utility_functions import load_conversations_by_id,write_log,save_metadata_to_json,gpt3_embeddings,timestamp_to_string

load_dotenv() #load environment variables
GREEN_COLOR = "#00EE00" # go
RED_COLOR = "#EE6363" # wait
GREY_COLOR = "#EEF6E6" # used for inout field
DARKER_GREY_COLOR = "#BEC4B8" #used for send button
VOICE_ID = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0" #female
ROLLING_MESSAGES_COUNT = 7 #can be increased for larger memory, at the cost of more tokens.
MESSAGES_FILEPATH = 'rolling_messages.txt' #create new file in root directory.
PROMPT_FILEPATH = 'prompt.txt' #create new file in root directory.
NUM_QUERY_RESULTS = 6 #number of relevant queries to inserted into prompt


class ChatBot:
    # Represents the chatbot using OpenAI API
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.openai_engine = 'text-davinci-003'
        self.messages = RollingMessages(MESSAGES_FILEPATH, ROLLING_MESSAGES_COUNT) #initializing message log object
        self.load_messages()
        self.openai_prompt = self.load_prompt()

    def load_messages(self):
        # Load the rolling messages log to be associated with chatbot.
        self.messages.load()

    def load_prompt(self):
        # Load the conversation prompt.
        with open(PROMPT_FILEPATH, 'r') as file:
            return file.read()

    def chat(self, prompt):
        # Send chat prompt to OpenAI and get the response.
        response = openai.Completion.create(
            engine=self.openai_engine,
            prompt=prompt,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()

    def generate_prompt(self, user_recent_text, relevant_convos):
        # Generate the conversation prompt with the latest user query.
        conversation = '\n'.join(self.messages.messages)
        prompt = self.openai_prompt.replace("<<PREVIOUS_CONVERSATION>>", conversation).replace("<<USER_RECENT_QUERY>>", user_recent_text).replace("<<RELEVANT_CONVERSATION>>", relevant_convos)
        return prompt




class BeckApp:
    # Represents the tkinter application for the Beck chatbot.

    def __init__(self, chatbot, text_to_speech, pinecone_api_key):
        self.chatbot = chatbot
        self.text_to_speech = text_to_speech
        self.is_listening = False
        self.root = tk.Tk()
        self.root.geometry("800x400")
        self.root.config(bg="white")
        self.root.resizable(False, False)
        self.root.title("Beck")
        
        #voice activation button
        self.button = tk.Button(self.root, text="", bg=GREEN_COLOR, fg=RED_COLOR, width=100, height=100,command=self.toggle_listening, relief='flat')
        self.button.place(relx=0.0, rely=0.5, anchor=tk.CENTER)
        # chat box
        self.message_box = tk.Text(self.root, bg="white", fg="black", height=23, width=54, relief='flat')
        self.message_box.place(relx=0.7225, rely=0.47, anchor=tk.CENTER)
        #scrollable message box
        self.scrollbar = tk.Scrollbar(self.root, command=self.message_box.yview)
        self.message_box.config(yscrollcommand=self.scrollbar.set) #link scrollbar
        self.message_box.config(state="disabled")
        # input field
        self.input_field = tk.Entry(self.root, width=58, bg=GREY_COLOR)
        self.input_field.place(relx=0.675, rely=0.96, anchor=tk.CENTER)
        self.input_field.bind("<Return>", self.send_message) # enter key bind to send message from input field
        #send button
        self.send_button = tk.Button(self.root, text="Send", width=8, height=1, command=self.send_message,bg=DARKER_GREY_COLOR, )
        self.send_button.place(relx=0.95, rely=0.96, anchor=tk.CENTER)
        # on close protocol
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)
        self.window_open = True #boolean for threading bug

        # Initialize pincone connection
        pinecone.init(api_key=pinecone_api_key, environment='northamerica-northeast1-gcp') # Establish connection to Pinecone
        self.vdb_index = pinecone.Index('beck') # Connect to index

        self.root.mainloop()

    def on_close(self):
        self.window_open = False #close window before thread finished, bug
        self.root.destroy()
        

    def toggle_listening(self):
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        self.is_listening = True
        self.button.config(state="disabled")  # Disable the button
        self.button.config(bg=RED_COLOR)
        self.root.update_idletasks()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.recognizer.pause_threshold = 1.5  # can be adjusted to the user's preference.
        self.listen_for_audio()

    def stop_listening(self):
        self.is_listening = False
        self.button.config(bg=GREEN_COLOR)
        self.root.update_idletasks()
        self.microphone = None
        self.recognizer = None
        self.button.config(state="normal")  # Enable the button

    def send_message(self, event=None):
        def process_message():
            user_input = self.input_field.get() # fetch input
            if user_input:
                self.input_field.delete(0, tk.END)
                self.message_box.tag_config("user", foreground="blue")
                self.message_box.tag_config("bot", foreground="green")

                # Update the message box with user message
                self.message_box.config(state="normal")
                self.message_box.insert(tk.END, "USER: " + user_input + "\n", "user")
                self.message_box.config(state="disabled")
                self.button.config(bg=RED_COLOR)
                self.root.update_idletasks()

                #generate prompt, call api
                #gather metadata
                payload = list() #tmp list for payload
                timestamp = time()
                timestring = timestamp_to_string(timestamp) # in 
                unique_id = str(uuid4())
                # create json doc wiht metadata (includes actual message)
                metadata = {'speaker': 'USER', 'time': timestamp, 'message': 'USER: ' + user_input, 'timestring': timestring, 'uuid': unique_id} # metadtata to be saved to local database
                # Save metadata to local database
                save_metadata_to_json(metadata,unique_id)
                #get embedding of user input
                user_input_embedding = gpt3_embeddings(user_input)
                #upload to payload(), will eventually be upserted to vector db. : (id, embedding)
                payload.append((unique_id, user_input_embedding))

                #now we must query for similar results from vector db
                results = self.vdb_index.query(vector=user_input_embedding, top_k=NUM_QUERY_RESULTS, include_values=False, include_metadata=False)
                relevant_convos = load_conversations_by_id(results)
                # GENERATE THE PROMPT
                prompt = self.chatbot.generate_prompt(user_recent_text=user_input, relevant_convos=relevant_convos)
                #get chatgpt response
                ai_response = self.chatbot.chat(prompt)
                #log for testing
                write_log(prompt,ai_response,unique_id)

                #gather metadata
                timestamp = time()
                timestring = timestamp_to_string(timestamp) # in 
                unique_id = str(uuid4())
                # create json doc wiht metadata (includes actual message)
                metadata = {'speaker': 'BECK', 'time': timestamp, 'message': 'BECK: ' + ai_response, 'timestring': timestring, 'uuid': unique_id}
                # Save metadata to local database
                save_metadata_to_json(metadata,unique_id)
                # create embedding of ai response
                ai_response_embedding = gpt3_embeddings(ai_response)
                #upload to payload(), will eventually be upserted to vector db. : (id, embedding)
                payload.append((unique_id, ai_response_embedding))
                # upsert new entries to vdb
                self.vdb_index.upsert(payload)
                if not self.window_open: #bug
                    return
                # Update the message box with bot response
                self.message_box.config(state="normal")
                self.message_box.insert(tk.END, "BECK: " + ai_response + "\n", "bot")
                self.message_box.config(state="disabled")
                self.root.update_idletasks()

                self.text_to_speech.speak(ai_response)
                if self.window_open==False:
                    return
                self.chatbot.messages.append(user_input, ai_response)
                if user_input.lower() in ["bye-bye", "bye", "goodbye"]:
                    self.on_close()
                self.stop_listening()

        # Create a new thread and start the process
        self.thread = threading.Thread(target=process_message)
        self.thread.start()

    def listen_for_audio(self):
        def process_audio():
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source)
                if not self.window_open:
                    return
                print("Recognizing...")
                try:
                    text = self.recognizer.recognize_google(audio)
                    # Configure the message tags for color highlighting
                    self.message_box.tag_config("user", foreground="blue")
                    self.message_box.tag_config("bot", foreground="green")
                    self.root.update_idletasks()
                    print("USER:", text)
                    # Update the message box with user and bot messages
                    self.message_box.config(state="normal")
                    self.message_box.insert(tk.END, "USER: " + text + "\n", "user")
                    self.root.update_idletasks()
        
                    #generate prompt, call api
                    #gather metadata
                    payload = list() #tmp list for payload
                    timestamp = time()
                    timestring = timestamp_to_string(timestamp) # in 
                    unique_id = str(uuid4())
                    # create json doc wiht metadata (includes actual message)
                    metadata = {'speaker': 'USER', 'time': timestamp, 'message': 'USER: ' + text, 'timestring': timestring, 'uuid': unique_id} # metadtata to be saved to local database
                    # Save metadata to local database
                    save_metadata_to_json(metadata,unique_id)
                    #get embedding of user input
                    user_input_embedding = gpt3_embeddings(text)
                    #upload to payload(), will eventually be upserted to vector db. : (id, embedding)
                    payload.append((unique_id, user_input_embedding))

                    #now we must query for similar results from vector db
                    results = self.vdb_index.query(vector=user_input_embedding, top_k=NUM_QUERY_RESULTS, include_values=False, include_metadata=False)
                    relevant_convos = load_conversations_by_id(results)
                    # GENERATE THE PROMPT
                    prompt = self.chatbot.generate_prompt(user_recent_text=text, relevant_convos=relevant_convos)
                    #get chatgpt response
                    ai_response = self.chatbot.chat(prompt)
                    #log for testing
                    write_log(prompt,ai_response,unique_id)

                    #gather metadata
                    timestamp = time()
                    timestring = timestamp_to_string(timestamp) # in 
                    unique_id = str(uuid4())
                    # create json doc wiht metadata (includes actual message)
                    metadata = {'speaker': 'BECK', 'time': timestamp, 'message': 'BECK: ' + ai_response, 'timestring': timestring, 'uuid': unique_id}
                    # Save metadata to local database
                    save_metadata_to_json(metadata,unique_id)
                    # create embedding of ai response
                    ai_response_embedding = gpt3_embeddings(ai_response)
                    #upload to payload(), will eventually be upserted to vector db. : (id, embedding)
                    payload.append((unique_id, ai_response_embedding))
                    # upsert new entries to vdb
                    self.vdb_index.upsert(payload)

                    if not self.window_open: #terminates window before thread stops 
                        return
                    print("BECK:", ai_response)
                    self.message_box.insert(tk.END, "BECK: " + ai_response + "\n", "bot")
                    self.message_box.config(state="disabled")
                    self.root.update_idletasks()

                    self.text_to_speech.speak(ai_response)
                    if not self.window_open:
                        return
                    self.chatbot.messages.append(text, ai_response)
                    if text in ["bye-bye", "bye", "goodbye"]:
                        self.on_close()

                except sr.UnknownValueError:
                    print("Sorry, I could not understand audio.")
                    self.text_to_speech.speak("Sorry, I could not understand audio.")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                    self.text_to_speech.speak("Could not request results from Google Speech Recognition service")
        
            self.stop_listening()

        # Create a new thread and start the text-to-speech process
        self.thread = threading.Thread(target=process_audio)
        self.thread.start()

# Driver code
if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    chatbot = ChatBot(openai.api_key)
    text_to_speech = TextToSpeech(VOICE_ID)
    app = BeckApp(chatbot, text_to_speech, pinecone_api_key)
