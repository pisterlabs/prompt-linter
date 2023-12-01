import openai
import os
from brian_utils import now
from config import openai_key as KEY
import brian_utils

openai.api_key = KEY

session_dirr = os.getcwd() + '/session_files'


def single_text_from_response(input_response):
    return input_response['choices'][0]['text']


def multiple_text_from_response(input_response):
    return [choice['text'] for choice in input_response['choices']]


class AlsGPT:
    def __init__(self, human_name, model_name="text-davinci-003", max_tokens=1024):

        self.model_name = model_name
        # intro is fed to the bot
        self.intro = f"""I am a highly, intelligent chat bot. I use English language. My name is Brian. The Human's name is {human_name}. I can remember things from earlier in the conversation."""
        self.conversation = self.intro
        self.max_tokens = max_tokens

        data = ''
        self.data_file = session_dirr + f'/{human_name}.txt'
        brian_utils.ensure_dir_and_file_exists(self.data_file)
        with open(self.data_file, 'r') as filey:
            data = filey.read()
            # print('read data from file: ', data)

        if data:
            self.conversation += ' ' + data
        self.conversation = self.conversation.replace(self.intro, '')  # this removes copies of the intro that happen in the file
        # print('conversation after replacing: ', self.conversation)
        self.conversation = self.intro + ' ' + self.conversation

    def summarise(self):
        prompt = self.conversation + f'\n Human: Please briefly summarize this conversation. \n Bot:'
        response = openai.Completion.create(engine=self.model_name, prompt=prompt, max_tokens=self.max_tokens, n=1, stop=None, temperature=0.5)
        bot_output = single_text_from_response(response)

        # creating a separate file for saving the pre-summary
        save_file = self.data_file.replace('.txt', str(now()) + '.txt')
        open(save_file, 'x')  # create file

        # save the old file
        with open(save_file, 'w') as filey:
            file_text = self.conversation + '\n~~~\n' + bot_output
            filey.write(file_text)
            print('done summarizing')
        self.conversation = bot_output
        return

    def say_to_bot(self, user_input):
        # summary run
        if 'summar' in user_input.lower() and 'conversation' in user_input.lower():
            self.summarise()
            say_to_user = 'Fully summarized internally.'
        else:
            # default conversation
            prompt = self.conversation + f'\n Human: {user_input} \n Bot:'
            # print('\n prompt:', prompt)
            response = openai.Completion.create(engine=self.model_name, prompt=prompt, max_tokens=self.max_tokens, n=1, stop=None, temperature=0.5)
            bot_output = single_text_from_response(response)
            print('\nBot:', bot_output)
            self.conversation = prompt + bot_output
            say_to_user = bot_output

        # print('\nconversation length: ', len(self.conversation), ' ~tokens:', len(self.conversation) // 4)
        if len(self.conversation) > 3000:
            say_to_user = say_to_user + ' Summarizing.'
            self.summarise()
        with open(self.data_file, 'w') as filey:
            filey.write(self.conversation)
            # print('read file data:\n', filey.read(), '\n')
        return say_to_user
