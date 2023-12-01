import discord
import os
from ChatGPT import ChatGPT
from LocalAI import LLM
from CommandInterpreter import CommandInterpreter
from Utils import send_msg_to_usr, constructHelpMsg
import asyncio

class PersonalAssistant:
    '''
    Personal assistant, interprets hard-coded and arbitrary user commands/messages
    '''
    def __init__(self, debug:bool):
        self.DEBUG = debug
        self.personal_assistant_channel = os.getenv('PERSONAL_ASSISTANT_CHANNEL')
        self.personal_assistant_state = None
        self.personal_assistant_modify_prompts_state = None
        self.personal_assistant_modify_prompts_buff = []
        self.personal_assistant_commands = {
            "help": "show this message",
            "pa_llama": "toggle the use of a llama model to interpret an unknown command (huge WIP)",
            "remind me": "format is `[remind me], [description], [numerical value], [time unit (s,m,h)]`; sets a reminder that will ping you in a specified amount of time",
            # 'shakespeare': 'generate a random snippet of shakespeare'
            'draw': "format is `[draw]; [prompt]` and it allows you to draw images using Dalle API from OpenAI, default is using Dalle3"
        }
        self.personal_assistant_command_options = self.personal_assistant_commands.keys()
        self.help_str = constructHelpMsg(self.personal_assistant_commands)
        self.cmd_prefix = "!"

        self.gpt_interpreter = ChatGPT(debug)
        prompt = "You are a personal assistant who interprets users requests into either one of the hard coded commands that you will learn about in a second or respond accordingly to the best of your knowledge."
        prompt += f"The hard-coded commands are: {str(self.personal_assistant_commands)}"
        prompt += f"If you recognize what the user wants, output a single line that will activate the hard coded command prefixed with {self.cmd_prefix} and nothing else. Otherwise, talk."
        self.gpt_interpreter._setPrompt(prompt)

        self.llama_pa_prompt = f"You are a virtual assistant agent discord bot. The available commands are {self.personal_assistant_commands}. Help the user figure out what they want to do. The following is the conversation where the user enters the unknown command. Output a one sentence response."
        self.llama_pa_toggle = False
        self.llama_interpreter = LLM('llama', '7B', self.llama_pa_prompt)

        self.command_interpreter = CommandInterpreter(self.help_str)

    async def main(self, msg : discord.message.Message) -> str:
        '''
        Handles the user input for one of the hard-coded commands, if unable to find a hard-coded command to fulfill request
        will use one of the LLM interpreters available (for now local LLAMA or chatgpt)

        Priority Order
        1. Hard coded commands
        2. ChatGPT Response -> Try hard coded, otherwise send back to user

        '''
        usr_msg = str(msg.content)

        # handle personal assistant state (if any)
        if self.personal_assistant_state == "modify prompts":
            return await self._pa_modify_prompts(self, usr_msg)

        # list all the commands available
        if usr_msg == f"{self.cmd_prefix}help":
            return self.help_str

        if usr_msg[0] == self.cmd_prefix:
            return await self.command_interpreter.main(msg, usr_msg[1:])

        # # check if user input is a hard-coded command
        # cmd = usr_msg.split(",")[0]
        # if cmd not in self.personal_assistant_command_options:
        #     if self.llama_pa_toggle:
        #         await send_msg_to_usr(msg, await self.llama_interpreter.generate(usr_msg))
        #     else:
        #         await send_msg_to_usr(msg, self.gpt_interpreter.main(msg, usr_msg))
        #     return

        # all commands below this are not case sensitive to the usr_msg so just lowercase it
        # usr_msg = usr_msg.lower()

        # toggle llama interpretting wrong commands instead of default gpt
        # idk why id want this, llama is dumb
        # if usr_msg == "pa_llama":
        #     if self.llama_pa_toggle: 
        #         self.llama_pa_toggle = False
        #         await send_msg_to_usr(msg, 'LLama interpret deselected.')
        #     else: 
        #         self.llama_pa_toggle = True
        #         await send_msg_to_usr(msg, 'LLama interpret selected.')
        #     return


        # # testing out nanoGPT integration
        # if usr_msg == "shakespeare":
        #     await send_msg_to_usr(msg, "Generating...")
        #     await send_msg_to_usr(msg, await self.local_gpt_shakespeare(length=100))
        #     return

        # let gpt interpret
        gpt_response = await self.gpt_interpreter.main(msg)
        if gpt_response[0] == self.cmd_prefix:
            return await self.command_interpreter.main(msg, gpt_response[1:])
        return gpt_response

        

    async def _pa_modify_prompts(self, usr_msg : str) -> str:
        '''
        handles changing the prompts for the personal assistant
        returns any message needed to be sent to user
        '''
        # user can cancel at any time
        if usr_msg == "cancel":
            # cancel modifying any prompts
            self.personal_assistant_state = None
            self.personal_assistant_modify_prompts_state = None
            self.personal_assistant_modify_prompts_buff = []
            return "Ok, cancelling."

        # Stage 1: usr picks a operator
        if self.personal_assistant_modify_prompts_state == "asked what to do":
            # check response
            if usr_msg == "edit":
                self.personal_assistant_modify_prompts_state = "edit"
                return "Ok which prompt would you like to edit? [enter prompt name]"
            elif usr_msg == "add":
                self.personal_assistant_modify_prompts_state = "add"
                return "Ok, write a prompt in this format: [name]<SEP>[PROMPT] w/o the square brackets."
            elif usr_msg == "delete":
                self.personal_assistant_modify_prompts_state = "delete"
                return "Ok, which prompt would you like to delete? [enter prompt name]"
            elif usr_msg == "changename":
                self.personal_assistant_modify_prompts_state = "changename"
                return "Ok, which prompt name would you like to rename? [enter prompt name]"
            else:
                return "Invalid response, please try again."

        # Stage 2: usr provides more info for an already chosen operator
        if self.personal_assistant_modify_prompts_state == "edit":
            self.personal_assistant_modify_prompts_buff.append(usr_msg)
            self.personal_assistant_modify_prompts_state = "edit2"
            return f"Ok, you said to edit {usr_msg}.\nSend me the new prompt for this prompt name. (just the new prompt in its entirety)"
        if self.personal_assistant_modify_prompts_state == "edit2":
            # update our mapping of prompt name to prompt dict, then write the new prompts to file
            prompt_name = self.personal_assistant_modify_prompts_buff.pop()
            new_prompt = usr_msg
            self.map_promptname_to_prompt[prompt_name] = new_prompt
            self.gpt_save_prompts_to_file() # write the new prompts to file
            self.personal_assistant_state = None
            self.personal_assistant_modify_prompts_state = None
            return f"Updated '{prompt_name}' to '{new_prompt}'"
        if self.personal_assistant_modify_prompts_state == "add":
            prompt_name = usr_msg.split("<SEP>")[0]
            prompt = usr_msg.split("<SEP>")[1]
            self.map_promptname_to_prompt[prompt_name] = prompt
            self.gpt_save_prompts_to_file() # write the new prompts to file
            self.personal_assistant_state = None
            self.personal_assistant_modify_prompts_state = None
            return f"Added '{prompt_name}' with prompt '{prompt}'"
        if self.personal_assistant_modify_prompts_state == "delete":
            prompt_name = usr_msg
            del self.map_promptname_to_prompt[prompt_name]
            self.gpt_save_prompts_to_file() # write the new prompts to file
            self.personal_assistant_state = None
            self.personal_assistant_modify_prompts_state = None
            return f"Deleted '{prompt_name}'"
        if self.personal_assistant_modify_prompts_state == "changename":
            self.personal_assistant_modify_prompts_buff.append(usr_msg)
            self.personal_assistant_modify_prompts_state = "changename2"
            return f"Ok, what would you like to change the {usr_msg} to?"
        if self.personal_assistant_modify_prompts_state == "changename2":
            prompt_name = self.personal_assistant_modify_prompts_buff.pop()
            new_prompt_name = usr_msg
            prompt = self.map_promptname_to_prompt[prompt_name]
            del self.map_promptname_to_prompt[prompt_name]
            self.map_promptname_to_prompt[new_prompt_name] = prompt
            self.gpt_save_prompts_to_file() # write the new prompts to file
            self.personal_assistant_state = None
            self.personal_assistant_modify_prompts_state = None
            return  f"Changed '{prompt_name}' to '{new_prompt_name}'"
