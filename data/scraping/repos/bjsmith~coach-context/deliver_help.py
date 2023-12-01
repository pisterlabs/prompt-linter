# create an app to deliver CBT to the client using the GPT-4 OpenAI API.
# with each session, the conversation should be summarized and saved
# in the next session the notes should be loaded so that the virtual therapist
# can pick up where they left off.
# the app should avoid counseling about serious issues that require a human therapist
# this is quite important.
import os
import pandas as pd
import json
import numpy as np
# Path: deliver_cbt.py
import openai
import time

from chat import CBTTerminal, ChatConfig, CoachingIOInterface, AsyncCoachingIOInterface
from dataclasses import dataclass
from storage_management import StorageManager,LocalStorageManagement

class CoachingSession:
    """
    manages the connections and the overall session
    """
    def __init__(self, frontend: CoachingIOInterface, channel_id, user_id, first_message=None, is_async=False, file_storage_manager: StorageManager = None):
        #self.session_id = session_id
        self.channel_id = channel_id
        self.user_id = user_id
        #self.notes = []
        
        self.notes_filename = "notes_" + str(self.user_id) + ".txt"
        self.snapshot_filename = "snapshot_" + str(self.user_id) + ".json"
        self.notes_filepath = "delivercbt_files/" + self.notes_filename
        self.snapshot_filepath = "delivercbt_files/" + self.snapshot_filename
        self.frontend = frontend
        self.is_ended= False
        #self.interview_count = None
        self.last_message_ts = None
        self.session_info = CoachingSessionInfo() #shared information that is available to the CoachingSession and the Therapist
        self.is_async = is_async

        #long-term, we should do the initialization as below serpately from the session
        self.therapist = EABTTherapist(
            user_id,
            get_ai_response_func=self.get_chatcompletion_response,
            session_info = self.session_info
            #send_message_func=self.frontend.send_message
            )
        self.initialize_openai_api_connection()
        if file_storage_manager is None: file_storage_manager = LocalStorageManagement('delivercbt_files/')
        self.file_storage_manager = file_storage_manager

        #setup
        self.setup_session(first_message)
        


    def setup_session(self, first_message=None):

        notes = self.get_notes(self.notes_filename)
        
        #this also sets the session number.

        #check to see if there was a snapshot last session
        snapshot_notes = self.check_for_snapshot()
        if snapshot_notes is not None:
            #there is an unclosed session state
            session_summary = self.therapist.summarize_session(snapshot_notes)
            self.save_notes(self.notes_filename, session_summary)
            if self.file_storage_manager.exists(self.snapshot_filename):
                self.file_storage_manager.remove(self.snapshot_filename)
                
            # load the therapist's notes again.
            notes = self.get_notes(self.notes_filename)

        self.session_info.notes = notes
        self.session_info.self_harm_instructions = self.get_self_harm_instructions()
        # start therapist, pass notes
        
        self.therapist.take_instructions(self.get_instructions())
        self.therapist.read_notes(notes)
        

        if first_message is not None:
            self.therapist.listen_to_client(first_message)

        if not self.is_async: #for asynchronous sessions, this is dealt with in its own async function
            #this therapist doesn't introduce themselves
            #because the client is already in the channel and has already said something
            # so the first message from the therapist should be the response to the client's message
            # combined with an appropriate introduction.
            introductory_text = self.therapist.open_session()
            if introductory_text != "": #the introductory text may be blank, if instead the therapist gives themselves an instruction to shape the response, rather than sending introductory text.
                self.frontend.send_message(introductory_text, channel_id=self.channel_id)
            response = self.therapist.respond()
            self.frontend.send_message(message= response, channel_id=self.channel_id)
            self.set_last_message_ts()

    def end_session(self):
        self.is_ended = True
        session_summary = self.therapist.summarize_session()
        self.save_notes(self.notes_filename, session_summary)
        #session is ending properly, we don't need to load the snapshot
        self.file_storage_manager.remove(self.snapshot_filename)
        return(None)

        
    def initialize_openai_api_connection(self):
        #load the API key from the YAML file "athentication.yml", under the item "openaikey"
        # use the yaml package

        config = ChatConfig.get_config()

        openai.api_key = config['openaikey']

    def set_last_message_ts(self,ts=None):
        if ts is None:
            ts = float(time.time())
        self.last_message_ts = ts

    async def respond_to_session_opening_async(self):
        await self.frontend.message_admin("Session opened for " + str(self.user_id) + " in channel " + str(self.channel_id))
        introductory_text = self.therapist.open_session()
        if introductory_text != "": #the introductory text may be blank, if instead the therapist gives themselves an instruction to shape the response, rather than sending introductory text.
            await self.frontend.send_message(introductory_text, channel_id=self.channel_id)
        response = self.therapist.respond()
        await self.frontend.send_message(message= response, channel_id=self.channel_id)
        self.set_last_message_ts()

        


    async def handle_message_async(self,message,ts):
        self.therapist.listen_to_client(message)
        self.frontend.indicate_response_coming()
        response = self.therapist.respond()
        await self.frontend.send_message(message= response, channel_id=self.channel_id)
        
        #self.set_last_message_ts()
        #now we are going to archive the session
        #this is different to saving notes
        #we archive a full copy of the session
        #that way if the session is ended abruptly,
        #we can write notes in the next session
        #then load the notes
        self.take_session_snapshot()


    def handle_message(self,message,ts):
        self.therapist.listen_to_client(message)
        self.frontend.indicate_response_coming()
        response = self.therapist.respond()
        self.frontend.send_message(message= response, channel_id=self.channel_id)
        
        self.set_last_message_ts()
        #now we are going to archive the session
        #this is different to saving notes
        #we archive a full copy of the session
        #that way if the session is ended abruptly,
        #we can write notes in the next session
        #then load the notes
        self.take_session_snapshot()

    def take_session_snapshot(self):
        #save the therapist notes in a python pickle file ready to restore for next time.
        #the notes are json so we can save it as a json file
        #we just need to save it with the user_id as the filename
        #so that we can load it next time
        
        json_to_save = self.therapist.messages


        self.file_storage_manager.save(json.dumps(json_to_save),self.snapshot_filename)
        # with open(self.snapshot_filename, 'w') as f:
        #     json.dump(json_to_save, f)

    def check_for_snapshot(self):
        #check to see if there was a snapshot last session
        #if there was, load it
        if self.file_storage_manager.exists(self.snapshot_filename):
        #if os.path.exists(self.snapshot_filename):
            # with open(self.snapshot_filename, 'r') as f:
            #     snapshot = json.load(f)
            snapshot_raw = self.file_storage_manager.load(self.snapshot_filename)
            snapshot = json.loads(snapshot_raw)
            #self.therapist.notes = notes
            #we don't want to load them as notes at this stage
            #delete the snapshot file
            
            return(snapshot)
        else:
            #no snapshot, so we start a new session
            return(None)
        

    def save_notes(self, notes_file, session_summary=None):
        #get the therapist to summarize the session
        if session_summary is None:
            session_summary = self.therapist.summarize_session()
        session_summary = (
            "Session " + str(self.session_info.interview_count) + ": " + 
            session_summary + "\n"
        )
        #save the therapist's notes, creating the file if it doesn't exist
        #and appending to it if it does
        self.file_storage_manager.open_append(session_summary, notes_file)
        

    def get_notes(self, notes_file):
        #load the therapist's notes
        if self.file_storage_manager.exists(notes_file):
            #notes is a text file with just a simple note. read it all in.
            notes = self.file_storage_manager.load(notes_file)
            #count the lines in the notes variable
            #the number of lines is the number of sessions
            #notes string is alreayd read out so we just need to count the lines
            self.session_info.interview_count = len(notes.splitlines()) +1
            # with open(notes_file, 'r') as f:
            #     notes = f.read()
            #     #get the number of lines of the file
            #     #the number of lines is the number of sessions
            #     #notes string is already read out so we just need to count the lines
            #     self.interview_count = len(notes.splitlines()) +1
            #     #add the number of sessions to the notes
                
                
                
        else:
            notes = "This is session 1. The therapist has never met the client before."
            self.session_info.interview_count = 1

        
        return(notes)


    def get_chatcompletion_response(self, messages_to_send, chat_model="gpt-3.5-turbo"):
        
        


        api_response = openai.ChatCompletion.create(
            model=chat_model,
            messages=messages_to_send
            )
        
        #process the response using json parsing, and extract the content
        #of the first choice

        

        text_response = api_response.choices[0].message['content']
        return text_response
    
    def get_instructions(self):
        return(self.get_prompt_from_flatfile(self.therapist.instruction_set_name))

    def get_self_harm_instructions(self):
        return(self.get_prompt_from_flatfile('anti_self_harm_prompt'))


    def get_prompt_from_flatfile(self, file_label):

        instructions_filepath = "delivercbt_files/" + file_label +".txt"

        with open(instructions_filepath, 'r') as f:
            instructions = f.read()

        return(instructions)

    
    def run(self):
        """
        Runs autonomously until the client exits.
        Only useful for handling a single client at this point.
        """
        session_in_progress = True
        


        # load the therapist's notes
        notes = self.get_notes(self.notes_filename)
        # start therapist, pass notes
        
        self.therapist.take_instructions(self.get_instructions())
        self.therapist.read_notes(notes)

        introductory_text = self.therapist.open_session()
        self.frontend.print_output(introductory_text)
    
        while session_in_progress:

            # solicit input from client
            client_input = self.frontend.get_input("")

            #probably a more AI friendly way to do this
            #like asking the AI engine whether the client is trying to exit
            #but this is a quick and dirty way to exit the session
            if client_input == "exit":
                session_in_progress = False
                break
            # pass to therapist and solicit response
            self.therapist.listen_to_client(client_input)


            self.frontend.indicate_response_coming()
            response = self.therapist.respond()

            # pass response to client
            self.frontend.print_output(response)
            # repeat
        # on exit, save notes
        self.save_notes(self.notes_filename)

@dataclass
class CoachingSessionInfo:
    """
    Stores various pieces of information about the session.
    An instantiation is made available to the therapist so that they can access the session info.
    """
    
    interview_count: int = None
    notes: str = None
    self_harm_instructions: str = None


class Therapist:
    """"
    manages the therapist's notes and the conversation
    """
    

    def __init__(self, session_id, get_ai_response_func,session_info: CoachingSessionInfo):
        #this is a set of information you build up before sending for a response
        #self.information_buffer = []
        self.messages = []
        self.get_ai_response = get_ai_response_func
        #self.notes = ""
        self.reminder_buffer = []
        self.instruction_set_name = 'alt_prompt'
        self.session_info = session_info

        self.initial_intro_text = "Hello, I'm a virtual therapist. I'm here to help you deal with your fears and anxieties using cognitive behavioral therapy. I am not equipped to talk with you about other psychological issues but we can talk about overcoming your fears. Please be aware a summary of this conversation will be recorded for training purposes."
        #self.subsequent_intro_text ="Hello again! Thank you for joining me today."

        
        pass

    def take_instructions(self, instructions):
        self.messages.append({"role": "system", "content": instructions})
        pass


    def self_monitor(self, therapist_talk):
        """
        this function is designed to monitor what the therapist is saying.
        If it needs to be corrected, it is designed to correct the therapist.
        Returns a boolean value indicating if a correction was made.
        """
        #count the number of words in "therapist_talk"
        #if the number of words is more than 100, they're monologuing too much.
        if len(therapist_talk.split()) > 100:
            print("reminded therapist to keep content short.")
            #self.reminder_buffer.append({"role": "system", "content": "(The therapist is talking too much. The therapist should ask more questions, and keep each reply to 100 words or less.)"})
            self.reminder_buffer.append({"role": "system", "content": "(Remember to keep each reply to 100 words or less.)"})
            return False
        
        return True



        



    def open_session(self):
                # this should be stored externally but this is a quick and dirty way to do it
        if self.session_info.interview_count == 1:
            introduction = self.initial_intro_text
        else:
            #introduction = self.subsequent_intro_text
            introduction = ""
            self.reminder_buffer.append({"role": "system", "content": "(A new session is beginning. In the first message, the therapist should briefly say hi, thank the client for joining them, and then respond to any substantive opening remarks from the client.)"})

        #if client has already said something, then we need to respond to that first.
        if np.any([m['role']=="user" for m in self.messages]):
            self.messages.append({
                "role": "system", "content": 
                "(The client has already started talking. " + 
                "After introducing themselves, the therapist should respond to the client's first statement" + 
                " and if appropriate, invite the client to tell them a little bit about themselves.)"})
        else:
            #basically this one runs if the therapist initiates the conversation
            auxillary_intro = "\nBefore we get started, would you like to tell me a bit about yourself?"
            introduction += auxillary_intro

        if introduction != "":
            self.messages.append({"role": "assistant", "content": introduction})


        
        return(introduction)

    def read_notes(self, notes):
        #self.information_buffer += "\n" + notes
        #self.notes = notes
        self.session_info.notes = notes
        self.take_instructions("The therapist's notes about previous sessions are as follows:\n" + notes)

    def listen_to_client(self, client_input):
        #self.information_buffer += "\n Client: " + client_input
        self.messages.append({"role": "user", "content": client_input})


    def get_unreplied_user_messages_in_stack(self, messages_to_send):
        last_messages = []
        for m in reversed(messages_to_send):
            if m['role'] == 'user':
                #this is the last message from the user
                last_messages.append(m)
            else:
                #we have reached the end of the user's messages
                #so we can stop iterating
                break

        return last_messages

    
    def check_for_self_harm_in_latest_messages(self):
        #get the last set of messages that are from the user
        #message_to_send is an array of dictionaries; start from the last item, and check if it is from the user
        #iterate backward until we get on that doesn't have role:user
        last_messages = self.get_unreplied_user_messages_in_stack(self.messages)

        #get just the text content of these messages
        last_messages_content = [m['content'] for m in last_messages]

        #get moderation ratings
        if len(last_messages_content) == 0:
            #no messages to check
            return(False)
        
        results_array = openai.Moderation.create(last_messages_content)
        self_harm_categories = ['self-harm','self-harm/instructions','self-harm/intent']
        for n in results_array['results']:
            for sfc in self_harm_categories: 
                #if the score is above 1%, then we have a match
                print(sfc + ": " + str(n['category_scores'][sfc]))
                if n['category_scores'][sfc]>0.01:
                    print('possible self harm detected.')
                    return(True)
        
        return(False)

    def respond(self):

        # send the information buffer to the AI and get a response
        if len(self.reminder_buffer) > 0:
            self.messages = self.messages + self.reminder_buffer
            self.reminder_buffer = []

        #chat_model="gpt-3.5-turbo"
        #chat_model = "gpt-3.5-turbo-1106"
        chat_model= 'gpt-4'
        #before we get a standard response, check for moderation.
        #we are only going to monitor self-harm at this stage.
        self_harm_detected = self.check_for_self_harm_in_latest_messages()
        if self_harm_detected:
            #if self harm detected, then we need to append an instruction to the bot to respond with a self harm warning with very specific information.
            self.take_instructions(self.session_info.self_harm_instructions)
            #use the latest model in order to ensure that the self harm warning is as accurate as possible
            chat_model= 'gpt-4'

        response = self.get_ai_response(self.messages, chat_model = chat_model)
        #now apply the self-monitoring function; it will get a new response if necessary.
        try_again = self.self_monitor(response)

        if try_again:
            response = self.get_ai_response(self.messages, chat_model = chat_model)
            
            # in the future this will need to be more sophisticated
            # to take into account multiple rounds of revision and reminders

        self.messages.append({"role": "assistant", "content": response})
        # clear the information buffer
        #self.information_buffer = []
        # send the response to the client
        return(response)
    
    def summarize_session(self, finish_messages = None):
        """
        summarizes the notes from the session
        by default, does this for the therapists messages
        But can also take a custom set of messages to summarize
        """
        if finish_messages is None:
            finish_messages = self.messages.copy()
        else:
            finish_messages = finish_messages.copy()
        #strip out system messages
        dialogue = [message for message in finish_messages if message["role"] != "system"]
        #now convert it from a list of dicts to a single string for summarization
        dialogue_string = ""
        for line in dialogue:
            if line["role"] == "user":
                dialogue_string += "\nClient: " + line["content"] + "\n"
            elif line["role"] == "assistant":
                dialogue_string += "\nTherapist: " + line["content"] + "\n"
        
        summary_request_string = "Please read the following conversation and then follow the instructions below it. \nBEGIN CONVERSATION:\n" + dialogue_string + "\nEND OF CONVERSATION.\n\nPlease summarize the above conversation, including summarizing the client's concerns and the progress that the therapist made with the client."
        summary_prompt = [{'role': 'user', 'content': summary_request_string}]
        response = self.get_ai_response(summary_prompt)
        
        return(response)

        

class CBTTherapist(Therapist):
    
    def __init__(self, session_id, get_ai_response_func,session_info: CoachingSessionInfo):
        super().__init__(session_id, get_ai_response_func, session_info)
        pass


class EABTTherapist(Therapist):

    def __init__(self, session_id, get_ai_response_func,session_info: CoachingSessionInfo):
        super().__init__(session_id, get_ai_response_func, session_info)

        self.initial_intro_text = "Hello, I'm a virtual therapist. I'm here to help you deal with your addictions and habits using emotional attachment behavioral therapy (EABT). I am not equipped to talk with you about other psychological issues but we can help you address addictions and unhealthy or undesired habits. Please be aware a summary of this conversation will be recorded for training purposes."
        self.instruction_set_name = 'eabt_prompt'
        pass



print("loading")



    


#start a session if this is the main file
if __name__ == "__main__":
    session = CoachingSession(1)
    session.run()
    