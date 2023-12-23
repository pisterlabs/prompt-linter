import openai
from openai import OpenAI
import os
import discord
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from Utils import constructHelpMsg
import time
from PIL import Image
import io, base64

class Dalle:
    def __init__(self, debug:bool):
        self.DEBUG = debug
        # openai.api_key = os.getenv("GPT3_OPENAI_API_KEY")
        self.model = "dall-e-3"
        self.client = OpenAI()

    # TODO: there's also option to allow editing of images only with DALLE2 model
        
    async def main(self, prompt : str) -> Image:
        '''
        Create an image using Dalle from openai and return it as a base64-encoded image
        '''
        def blocking_api_call():
            return self.client.images.generate(
                        model = self.model,
                        prompt = prompt,
                        size = "1024x1024",
                        quality = "standard",
                        response_format = "b64_json",
                        n = 1,
                    )

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(executor, blocking_api_call)

        # decode from base 64 json into image
        encoded_img = response.data[0].b64_json
        image = Image.open(io.BytesIO(base64.b64decode(encoded_img)))

        # return image
        return image


class ChatGPT:
    def __init__(self, debug:bool):
        self.DEBUG = debug

        self.client = OpenAI()
        self.gpt3_channel_name = os.getenv('GPT3_CHANNEL_NAME')
        self.gpt3_channel_id = os.getenv('GPT3_CHANNEL_ID')
        self.gpt3_model_to_max_tokens = {
            "gpt-4-1106-preview": [128000, "Apr 2023"], 
            "gpt-4-vision-preview" : [128000, "Apr 2023"], 
            "gpt-4" : [8192, "Sep 2021"]
        }
        self.gpt3_settings = {
            "model": ["gpt-4-vision-preview", "str"], 
            "prompt": ["", "str"],
            "messages" : [[], "list of dicts"],
            "temperature": ["0.0", "float"],
            "top_p": ["1.0", "float"],
            "frequency_penalty": ["0", "float"],
            "presence_penalty": ["0", "float"],
            "max_tokens": [128000, "int"],
        }
        self.chatgpt_name="assistant"
        self.cmd_prefix = "!"

        # gpt prompts
        self.gpt_prompts_file = os.getenv("GPT_PROMPTS_FILE") # pickled prompt name -> prompts dict
        self.all_gpt3_available_prompts = None # list of all prompt names
        self.map_promptname_to_prompt = None # dictionary of (k,v) = (prompt_name, prompt_as_str)
        self.curr_prompt_name = None  # name of prompt we're currently using

        self.commands = {
            "help" : "display this message",
            "convo len" : 'show current gpt3 context length',
            "reset thread" : 'reset gpt3 context length',
            "show thread" : 'show the entire current convo context',
            "gptsettings" : 'show the current gpt3 settings',
            # "gptreplsettings" : 'show gpt3 repl settings',
            "gptset": "format is `gptset, [setting_name], [new_value]` modify gpt3 settings",
            # "gptreplset": "format is `gptreplset, [setting_name], [new_value]` modify gpt3 repl settings",
            "curr prompt": "get the current prompt name",
            "change prompt": "format is `change prompt, [new prompt]`, change prompt to the specified prompt(NOTE: resets entire message thread)",
            "show prompts": "show the available prompts for gpt3",
            "list models": "list the available gpt models",
            "modify prompts": "modify the prompts for gpt",
            "save thread": "save the current gptX thread to a file",
            "show old threads": "show the old threads that have been saved",
            "load thread": "format is `load thread, [unique id]` load a gptX thread from a file",
            "delete thread": "format is `delete thread, [unique id]` delete a gptX thread from a file",
            "current model": "show the current gpt model",
            "swap": "swap between gpt3.5 and gpt4 (regular)",
        }
        self.commands_help_str = constructHelpMsg(self.commands)

        # initialize prompts
        self.gpt_read_prompts_from_file() # read the prompts from disk
        self.curr_prompt_name = self.all_gpt3_available_prompts[0] # init with first prompt
        self.gpt_context_reset()
    
    async def testFunc(self, msg : discord.message.Message) -> None:
        '''
        this function exists as a point to test out new features 
        '''
        print("testFunc")

    async def gen_gpt_response(self, msg : discord.message.Message, settings_dict: dict = None) -> str:
        '''
        retrieves a GPT response given a string input and a dictionary containing the settings to use
        returns the response str
        '''
        if settings_dict is None:
            settings_dict = self.gpt3_settings

        # init content with the user's message
        content = [
            {"type": "text",
             "text": msg.content
            }
        ]

        # attach images if present
        if msg.attachments:
            if settings_dict["model"][0] == "gpt-4-vision-preview":
                # add the image urls to the content
                for attachment in msg.attachments:
                    image_dict = {"type": "image_url"}
                    image_dict["image_url"] = attachment.url
                    content.append(image_dict)
            else:
                if self.DEBUG: print(f"DEBUG: tried to attach image to non-vision model, abortting request.")
                return "This model does not support images. Request aborted."

        new_usr_msg = {
            "role": "user",
            "content": content
        }

        if self.DEBUG: print(f"DEBUG: {new_usr_msg=}")

        ##############################
        # update list of messages, then use it to query
        settings_dict["messages"][0].append(new_usr_msg)

        def blocking_api_call():
            # query
            return self.client.chat.completions.create(
                model = settings_dict["model"][0],
                messages = settings_dict["messages"][0],
                temperature = float(settings_dict["temperature"][0]),
                top_p = float(settings_dict["top_p"][0]),
                frequency_penalty = float(settings_dict["frequency_penalty"][0]),
                presence_penalty = float(settings_dict["presence_penalty"][0]),
                max_tokens = 4096
            )
        
        # Run the blocking function in a separate thread using run_in_executor
        if self.DEBUG: print(f"DEBUG: Sent to ChatGPT API: {settings_dict['messages'][0]}")
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            completion = await loop.run_in_executor(executor, blocking_api_call)
        
        response_msg = completion.choices[0].message.content
        if self.DEBUG: print(f"DEBUG: Got response from ChatGPT API: {response}")
        return response_msg

    ################# Entrance #################
    async def main(self, msg : discord.message.Message) -> str:
        '''
        Entrance function for all ChatGPT API things.
        Either modifies the parameters or generates a response based off of current context and new user message.
        Returns the generation.
        '''
        usr_msg = str(msg.content)
        # catch if is a command
        if usr_msg[0] == self.cmd_prefix:
            # pass to PA block without the prefix
            return await self.modifyParams(msg, usr_msg[1:])

        # check to see if we are running out of tokens for current msg log
        # get the current thread length
        curr_thread = await self.get_curr_gpt_thread()
        curr_thread_len_in_tokens = len(curr_thread) / 4 # 1 token ~= 4 chars
        while curr_thread_len_in_tokens > int(self.gpt3_settings["max_tokens"][0]):
            # remove the 2nd oldest message from the thread (first oldest is the prompt)
            self.gpt3_settings["messages"][0].pop(1)
        
        # use usr_msg to generate new response from API
        gpt_response = await self.gen_gpt_response(msg)

        # reformat to put into messages list for future context, and save
        formatted_response = {"role":self.chatgpt_name, "content":gpt_response}
        self.gpt3_settings["messages"][0].append(formatted_response)

        return gpt_response
        
    ################# Entrance #################

    def _setPrompt(self, prompt : str) -> None:
        '''
        set the prompt for the model and update the messages settings dict
        '''
        self.gpt3_settings["prompt"][0] = prompt
        l = self.gpt3_settings["messages"][0]
        if len(l) == 0:
            l.append([{'role':'assistant', 'content':prompt}])
        else:
            l[0] = {'role':'assistant', 'content':prompt}
        self.gpt3_settings["messages"][0] = l

    async def modifyParams(self, msg : discord.message.Message, usr_msg : str) -> str:
        '''
        Modifies ChatGPT API params.
        Returns the output of an executed command or returns an error/help message.
        '''
        # convert shortcut to full command if present
        usr_msg = self.shortcut_cmd_convertor(usr_msg)

        # help
        if usr_msg == "help": return self.commands_help_str

        # save current msg log to file 
        if usr_msg == "save thread":
            global time
            # pickle the current thread from gptsettings["messages"][0]
            msgs_to_save = self.gpt3_settings["messages"][0]
            # grab current time in nanoseconds
            curr_time = time.time()
            # pickle the msgs_to_save and name it the current time
            with open(f"./pickled_threads/{curr_time}.pkl", "wb") as f:
                pickle.dump(msgs_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            return f"Saved thread to file as {curr_time}.pkl"
 
        # show old threads that have been saved
        if usr_msg == "show old threads":
            ret_str = ""
            # for now, list all the threads...
            for filename in os.listdir("./pickled_threads"):
                # read the file and unpickle it
                with open(f"./pickled_threads/{filename}", "rb") as f:
                    msgs_to_load = pickle.load(f)
                    ret_str += f"Thread id: {filename[:-4]}\n" # hide the file extension when displayed, its ugly
                    for tmp in msgs_to_load:
                        tmp_role = tmp["role"]
                        tmp_msg = tmp["content"]
                        ret_str += f"###{tmp_role.capitalize()}###\n{tmp_msg}\n###################\n"
                    ret_str += f"{'~ '*30}"
            return ret_str

        # load msg log from file
        if usr_msg[:11] == "load thread":
            tmp = usr_msg.split(",")
            if len(tmp) != 2:
                return "No thread id specified. usage: [load thread, THREAD_ID]"

            thread_id = tmp[1].strip()

            if len(thread_id) == 0:
                return "No thread id specified"

            if thread_id[-4:] == ".pkl":
                thread_id = thread_id[:-4]

            # read the file and unpickle it
            with open(f"./pickled_threads/{thread_id}.pkl", "rb") as f:
                msgs_to_load = pickle.load(f)
                # set the current gptsettings messages to this 
                self.gpt3_settings["messages"][0] = msgs_to_load
            return  f"Loaded thread {thread_id}.pkl"
        
        # delete a saved thread
        if usr_msg[:13] == "delete thread":
            thread_id = usr_msg.split(",")[1].strip()

            if len(thread_id) == 0:
                return "No thread id specified"

            # delete the file
            os.remove(f"./pickled_threads/{thread_id}.pkl")
            return f"Deleted thread {thread_id}.pkl"

        # list available models of interest
        if usr_msg == "list models":
            tmp = "".join([f"{k}: {v}\n" for k,v in self.gpt3_model_to_max_tokens.items()])
            ret_str = f"Available models:\n{tmp}" 
            if self.DEBUG: print(f"DEBUG: !list models\n {tmp}")
            return ret_str

        # show the current gpt3 prompt
        if usr_msg == "curr prompt":
            return self.curr_prompt_name

        # just show current model
        if usr_msg == "current model":
            return f"Current model: {self.gpt3_settings['model'][0]}"
        
        # toggle which model to use (toggle between the latest gpt4 turbo and the vision model)
        if usr_msg == "swap":
            curr_model = self.gpt3_settings["model"][0]
            if self.DEBUG: print(f"DEBUG: swap: {curr_model=}")
            if curr_model == "gpt-4-vision-preview":
                await self.modifygptset(msg, "gptset model gpt-4-1106-preview")
            else:
                await self.modifygptset(msg, "gptset model gpt-4-vision-preview")
            return f'Set to: {self.gpt3_settings["model"][0]}'

        # add a command to add a new prompt to the list of prompts and save to file
        if usr_msg == "modify prompts":
            if self.personal_assistant_state is None:
                self.personal_assistant_state = "modify prompts"
                self.personal_assistant_modify_prompts_state = "asked what to do" 
                return f"These are the existing prompts:\n{self.get_all_gpt_prompts_as_str()}\nDo you want to edit an existing prompt, add a new prompt, delete a prompt, or change a prompt's name? (edit/add/delete/changename)"

        # change gpt3 prompt
        if usr_msg[:13] == "change prompt":
            # accept only the prompt name, update both str of msgs context and the messages list in gptsettings
            try:
                self.curr_prompt_name = list(map(str.strip, usr_msg.split(',')))[1]
                self.gpt_context_reset()
                return "New current prompt set to: " + self.curr_prompt_name
            except Exception as e:
                return "usage: change prompt, [new prompt]"

        # show available prompts as (ind. prompt)
        if usr_msg == "show prompts":
            return self.get_all_gpt_prompts_as_str()

        # show user current GPT3 settings
        if usr_msg == "gptsettings":
            return self.gptsettings()

        # user wants to modify GPT3 settings
        if usr_msg[0:6] == "gptset":
            await self.modifygptset(msg, usr_msg)
            return self.gptsettings()
        
        # show the current thread
        if usr_msg == "show thread":
            ret_str = ""
            for tmp in self.gpt3_settings["messages"][0]:
                tmp_role = tmp["role"]
                tmp_msg = tmp["content"]
                ret_str += f"###{tmp_role.capitalize()}###\n{tmp_msg}\n###################\n"
            return ret_str

        # reset the current convo with the curr prompt context
        if usr_msg == "reset thread":
            self.gpt_context_reset()
            return f"Thread Reset. {await self.get_curr_convo_len_and_approx_tokens()}"
        
        # check curr convo context length
        if usr_msg == "convo len":
            return await self.get_curr_convo_len_and_approx_tokens()
        
        return "Unknown command."

    def shortcut_cmd_convertor(self, usr_msg :str) -> str:
        '''
        if the user enters a shortcut command, convert it to the actual command
        '''
        if usr_msg == "rt":
            return "reset thread"
        if usr_msg == "cl":
            return "convo len"
        if usr_msg == "st": 
            return "show thread"
        if usr_msg[:2] == "cp":
            return "change prompt" + usr_msg[1:]
        if usr_msg == "save":
            return "save thread"
        if usr_msg[:4] == "load" and usr_msg[5:11] != "thread":
            return "load thread" + usr_msg[3:]
        if usr_msg == "lm":
            return "list models"
        if usr_msg == "cm":
            return "current model"

        # not a shortcut command
        return usr_msg

    # convo len
    async def get_curr_convo_len_and_approx_tokens(self) -> str:
        '''
        returns a string of the current length of the conversation and the approximate number of tokens
        '''
        tmp = len(await self.get_curr_gpt_thread())
        return f"len:{tmp} | tokens: ~{tmp/4}"
    
    # changing gptsettings
    async def modifygptset(self, msg : discord.message.Message, usr_msg : str) -> None:
        ''' 
        Executes both gptset and gptsettings (to print out the new gpt api params for the next call)
        expect format: gptset [setting_name] [new_value]

        Returns None if ok, else returns a error msg string.
        '''
        try:
            self.gptset(usr_msg, self.gpt3_settings)
        except Exception as e:
            return "gptset: gptset [setting_name] [new_value]"
        return None
    
    def gpt_save_prompts_to_file(self) -> None:
        '''
        saves the prompt_name -> prompt dictionary to disk via pickling
        '''
        with open(self.gpt_prompts_file, "wb") as f:
            pickle.dump(self.map_promptname_to_prompt, f, protocol=pickle.HIGHEST_PROTOCOL)

    def gpt_read_prompts_from_file(self) -> None:
        '''
        reads all the prompts from the prompt file and stores them in self.all_gpt3_available_prompts and the mapping
        '''
        # reset curr state of prompts
        self.all_gpt3_available_prompts = [] # prompt names
        self.map_promptname_to_prompt = {} # prompt name -> prompt

        # load in all the prompts
        with open(self.gpt_prompts_file, "rb") as f:
            # load in the pickled object
            self.map_promptname_to_prompt = pickle.load(f)
            # get the list of prompts
            self.all_gpt3_available_prompts = list(self.map_promptname_to_prompt.keys())

    def gpt_context_reset(self) -> None:
        '''
        resets the gpt3 context
        > can be used at the start of program run and whenever a reset is wanted
        '''
        self.gpt3_settings["messages"][0] = [] # reset messages, should be gc'd
        self.gpt3_settings["messages"][0].append({"role":self.chatgpt_name, "content":self.map_promptname_to_prompt[self.curr_prompt_name]})
    
    async def get_curr_gpt_thread(self) -> str:
        '''
        generates the current gpt conversation thread from the gptsettings messages list
        '''
        ret_str = ""
        for msg in self.gpt3_settings["messages"][0]:
            ret_str += f"{msg['role']}: {msg['content']}\n" 
        return ret_str

    def gptsettings(self) -> str:
        '''
        returns the available gpt3 settings, their current values, and their data types
        excludes the possibly large messages list
        '''
        gpt3_settings = self.gpt3_settings
        return "".join([f"{key} ({gpt3_settings[key][1]}) = {gpt3_settings[key][0]}\n" for key in gpt3_settings.keys() if key != "messages"])

    def gptset(self, usr_msg : str, options : str = None) -> None:
        '''
        format is 

        GPTSET, [setting_name], [new_value]

        sets the specified gpt3 parameter to the new value
        
        e.g.
        usr_msg = prompt, "bob the builder loves to build"

        # FIXME: allow prompt editing
        '''
        tmp = usr_msg.split()
        setting, new_val = tmp[1], tmp[2]
        self.gpt3_settings[setting][0] = new_val # always gonna store str

        # if setting a new model, update the max_tokens
        if setting == "model":
            x = self.gpt3_model_to_max_tokens[new_val] # (max_tokens, date of latest date)
            self.gpt3_settings["max_tokens"][0] = x[0]

    def get_all_gpt_prompts_as_str(self):
        '''
        constructs the string representing each [prompt_name, prompt] as one long string and return it
        '''
        return "".join([f"Name: {k}\nPrompt:{v}\n----\n" for k,v in self.map_promptname_to_prompt.items()])

