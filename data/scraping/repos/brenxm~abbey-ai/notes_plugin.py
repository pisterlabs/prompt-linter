from dotenv import load_dotenv
from response_streamer import quick_prompt_response
import os
import json
import datetime
import openai
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Notes:
    """
    This class handles operations related to notes: Load, Retrieve, Create, Update, Delete and Get all notes.
    Each note object follows the schema:
    {
        "title": str,
        "content": str,
        "metadata": {
            "date_created": str,
            "last_opened": str,
            "last_updated": str
        }
    }
    """

    def __init__(self, chat_history, **kwarg):
        self.chat_history = chat_history
        self.notes_file_path = os.path.join("data", "notes_data.json")
        self.load()
        self.parser_obj = {
                "keywords": re.compile(r"((my notes|my plans).+?(check|update|modify|add|create|make|write|read|open|delete|remove)|(update|modify|add|create|make|write|read|open|check|delete|remove).+?(my notes|my plans))", re.IGNORECASE),
                "prior_function": [
                    {
                        "function": self.prior_fn,
                        "arg": {

                        }
                    }
                ]
            }
        

    def load(self):
        """
        Load notes data from JSON file. If file doesn't exist, create an empty one.
        """
        try:
            with open(self.notes_file_path, "r") as f:
                data = f.read()
                data_list = json.loads(data)
                self.notes = data_list
        except:
            self.save([])

    def save(self, data):
        """
        Save notes data to JSON file.
        """
        with open(self.notes_file_path, "w") as f:
            json.dump(data, f, indent=4)

    def read_note(self, prompt_input, title = False):
        """
        Returns content of a note matching the given title.
        """
        if not title or len(self.notes) == 0:
            return {
                "messages": [
                    {
                        "role": "system", "content": "You tried looking for that note but that note is not existing"
                    }
                ]
            }
        for note_obj in self.notes:
            if note_obj["title"] == title:
                return {
                    "messages": [
                        {
                            "role": "user", "content": prompt_input
                        },
                        {
                            "role": "system", "content": f"Retrieved note {note_obj['content']}"
                        }
                        ]
                    }
        
        return {
                "messages": [
                    {
                        "role": "system", "content": "You tried looking for that note but that note is not existing"
                    }
                ]
            }

    def new_note(self, title, content, prompt_input):
        """
        Creates a new note with current date and time as metadata.
        """
        note_obj = {"title": title, "content": content}
        current_time = str(datetime.datetime.now())
        note_obj["metadata"] = {
            "date_created": current_time,
            "last_opened": current_time,
            "last_updated": current_time,
        }

        self.notes.append(note_obj)
        self.save(self.notes)
        return {
            "messages": [
                {
                    "role": "user", "content": prompt_input
                },
                    {"role": "system", "content": f"You (assistant) create a note titled {title}"},
                        ],

                
        }

    def update_note(self, title, prompt_input):
        """
        Updates content of an existing note and its last updated time in metadata.
        """
        note_data = [note["content"] for note in self.notes if note["title"] == title][0]
        if title not in self.get_titles():
            return {
                "messages": [
                    {"role": "system", "content": f"You tried looking for file {title}, but it's not in the notes."}
                ]
            }

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = [
                 {
                        "role": "system", "content": f"current date and time: {str(datetime.datetime.now())}"
                },
                {
                    "role": "system", "content": f"The title of the note is '{title}' and the content of the note is '{note_data}'"
                },
                {
                    "role": "user", "content": prompt_input 
                }
            ],
            functions = [
               {
                   "name": "update_note",
                   "description": "Update the note by replacing the content with new content",
                   "parameters": {
                       "type": "object",
                       "properties": {
                           "content": {
                               "type": "string",
                               "description": "The content that will replace the old content of the note"
                           }
                       }
                   },
                   "required": ["content"]
               }
            ],

            function_call = {"name": "update_note"}
        )
        
        print(response)
        if response["choices"][0]["message"]["function_call"]:
            arg = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
            content = arg.get("content")
            for note_obj in self.notes:
                if note_obj["title"] == title:
                    note_obj["content"] = content
                    note_obj["metadata"]["last_updated"] = str(datetime.datetime.now())

            self.save(self.notes)
            return {
                "messages": [
                    {
                        "role": "user", "content": prompt_input
                    },
                    {"role": 'system', "content": "In the background, You (assistant) completed this task"},
                ]
            }
        
        else:
            raise("There was an error")
        


    def delete_note(self, title, prompt_input):
        """
        Deletes a note.
        """
        if len(self.notes) == 0:
            return {
                "messages": [
                    {"role": "user", "content": prompt_input},
                    {"role": "system", "content": "You (assistant) tried deleting it but there is no note to delete"}
                ]
            }
        
        self.notes = [note_obj for note_obj in self.notes if note_obj["title"] != title]
        self.save(self.notes)
        return {
            "messages": [
                {
                    "role": "user", "content": prompt_input
                },
                {
                    "role": "system", "content": f"You deleted the note titled '{title}' as requested"
                },
            ]
        }

    def get_all_notes(self):
        """
        Returns a string of all the notes in the JSON file.
        """
        return json.dumps(self.notes)

    def get_titles(self):
        """
        Returns a list of titles (strings) of available notes
        """
        result = [notes_obj["title"] for notes_obj in self.notes]
        if len(result) == 0:
            return "No note in notes"
        return result
            
    
    def list_of_notes(self, obj):

        titles = self.get_titles()
        if isinstance(titles, list):
            data = ", ".join(titles)
        
        else:
            data = titles
        

        response_obj = {}
        response_obj["messages"] = [
            {
            "role": "system", "content": f"You (Assistant) obtained the list of notes from the system. The list are [{data}]"
            }
        ]

        return response_obj

    def prior_fn(self, obj):
        quick_system_response = f"The user ask you to complete a task. This is the requested task '{obj['prompt']}'. Response a quick, laconic confirmation."

        response_obj = obj
        response_obj["messages"] = []
        function_map = {
            "read": self.read_note,
            "create": self.new_note,
            "update": self.update_note,
            "delete": self.delete_note,
                
        }
        keyword_used = obj["keyword_obj"]["keyword_used"].lower()
        
        
        # Get keyword
        keyword_dict = {
            "read": ["read", "open", "check"],
            "create": ["make", "write", "create"],
            "update": ["update", "modify"],
            "delete": ["remove", "delete"],
            "titles": ["list of notes"]
        }
        
        # Iterate over each keyword in keyword used 
        action_key = ""
        for word in keyword_used.split(" "):
            for key, keywords in keyword_dict.items():
                if word in keywords:
                    action_key = key

        
        if action_key == 'read':
            response = openai.ChatCompletion.create(
                model = 'gpt-4',
                messages = [
                    {
                        "role": "system", "content": f"These are the chat history. {str(self.chat_history)}"
                    },
                    {
                        "role": "system", "content": f"current date and time: {str(datetime.datetime.now())}"
                    },
                    {
                        "role": "system", "content": f"Available titles: [{self.get_titles()}], 'none' if not in the titles"
                    },
                    {
                        "role": "user", "content": obj["prompt"]
                    }
                ],
                
                functions = [{
                        "name": "read",
                        "description": "Get the content of the notes or plans",
                        "parameters":{
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "The title of the note or plan saved. Use only the given titles, if the plan is not existing, title should be a string of 'none'"
                                }
                            },
                            "required": ["title"]
                        }
                    }
                ],
                
                function_call = {"name": "read"}
            )
            
        elif action_key == "create":
            quick_prompt_response(quick_system_response)
            response = openai.ChatCompletion.create(
                model = 'gpt-4',
                messages = [
                    {
                        "role": "system", "content": f"These are the chat history. {str(self.chat_history)}"
                    },
                    {
                        "role": "system", "content": f"current date and time: {str(datetime.datetime.now())}"
                    },
                    {
                        "role": "user", "content": obj["prompt"]
                    }
                ],
                
                functions = [{
                        "name": "create",
                        "description": "Create a new item in the notes\\plans",
                        "parameters":{
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "The title of the note or plan to be saved."
                                },
                                "content": {
                                    "type": "string",
                                    "description": "content/description/body of the new note"
                                }
                            },
                            "required": ["title", "content"]
                        }
                    }
                ],
                
                function_call = {"name": "create"}
            )
             
        elif action_key == "update":
            quick_prompt_response(quick_system_response)
            response = openai.ChatCompletion.create(
                
                model = 'gpt-4',
                messages = [
                    {
                        "role": "system", "content": f"These are the chat history. {str(self.chat_history)}"
                    },
                     {
                        "role": "system", "content": f"current date and time: {str(datetime.datetime.now())}"
                    },
                    {
                        "role": "system", "content": f"Title of all notes: {self.get_titles()}"
                    },
                    {
                        "role": "user", "content": obj["prompt"]
                    }
                ],
                
                functions = [{
                        "name": "update",
                        "description": "Update an existing file in the notes",
                        "parameters":{
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "The title of the note or plan to be updated"
                                },
                            },
                            "required": ["title",]
                        }
                    }
                ],
                
                function_call = {"name": "update"}
            )
             
        elif action_key == "delete":
            titles = self.get_titles()
            
            response = openai.ChatCompletion.create(
                model = 'gpt-4',
                messages = [
                    {
                        "role": "system", "content": f"These are the chat history. {str(self.chat_history)}"
                    },
                     {
                        "role": "system", "content": f"current date and time: {str(datetime.datetime.now())}"
                    },
                    {
                        "role": "system", "content": f"Available titles: [{' '.join(titles)}]"
                    },
                    {
                        "role": "user", "content": obj["prompt"]
                    }
                ],
                
                functions = [{
                        "name": "delete",
                        "description": "Delete an existing file\\items in the notes",
                        "parameters":{
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "The title of the note or plan to be deleted"
                                },
                            },
                            "required": ["title"]
                        }
                    }
                ],
                
                function_call = {"name": "delete"}
            )

        fn_name = response["choices"][0]["message"]["function_call"]["name"]
        fn = function_map.get(fn_name)
        
        args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
        args["prompt_input"] = obj["prompt"]
        fn_response = fn(**args)  # Pass the updated args as parameters to the function
        
        if "messages" in fn_response:
            response_obj["messages"] += fn_response["messages"]

        if "histories" in fn_response:
            response_obj["histories"] = fn_response["histories"]

        response_obj["delete_prompt"] = True
        return response_obj
    
    
def register(arg):
    return Notes(**arg)