from pandas import json_normalize  
import pandas as pd
import sys
sys.path.append(r"C:\Users\silvh\OneDrive\lighthouse\custom_python")
sys.path.append(r"C:\Users\silvh\OneDrive\lighthouse\portfolio-projects\online-PT-social-media-NLP\src")
from silvhua import *
from datetime import datetime
# from pypdf import PdfReader
import pandas as pd
import openai
import os
# from openai.embeddings_utils import get_embedding, cosine_similarity
import re
env_name = 'api_openai'

# See the 2023-04-6 notebook (iteration 11) for usage.
class Chatbot:
    """
    Requrired paramaters:
        - text (str): Text to feed to GPT for summarization.

    Optional parameters
        - system_role (str): ChatGPT parameter. 
            Default is "You are an expert at science communication."
        - temperature (float): ChatGPT parameter. Default is 0.7.
        - n_choices (int): Number of ChatGPT responses to generate. Default is 5.
        - max_tokens (int): Token limit for ChatGPT response.
        - model (str): ChatGPT model to use. Default is "gpt-3.5-turbo".
    """
    def __init__(self, text, system_role, model, temperature, n_choices, max_tokens):
        self.text = text
        self.system_role = system_role
        self.temperature = temperature
        self.n_choices = n_choices
        self.max_tokens = max_tokens
        self.model = model
    
    def create_prompt(self, task):
        system_role = f'{self.system_role}'
        user_input = f"""Given the following text: {self.text} \n {task}"""
        messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_input},]

        print('\tDone creating prompt')
        # print(messages)
        return messages

    def gpt(self, messages):
        print('\tSending request to GPT-3')
        print(f'\t\tRequesting {self.n_choices} choices using {self.model}')
        openai.api_key = os.getenv('api_openai')
        response = openai.ChatCompletion.create(
            model=self.model, messages=messages, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens,
            n=self.n_choices
            )
        print('\tDone sending request to GPT-3')
        return response


def reply(task, text, model="gpt-3.5-turbo", temperature=0.7, n_choices=5, max_tokens=1000,
          text_key=['text_discussion'],
        system_role = "You are an expert at science communication."
        ):
    """
    Send a ChatCompletion request to ChatGPT via the Chatbot class.

    Requrired paramaters:
        - task (str): Task to include in ChatGPT prompt.
        - text (str): Text to feed to GPT for summarization.

    Optional parameters
        - system_role (str): ChatGPT parameter. 
            Default is "You are an expert at science communication."
        - temperature (float): ChatGPT parameter. Default is 0.7.
        - n_choices (int): Number of ChatGPT responses to generate. Default is 5.
        - max_tokens (int): Token limit for ChatGPT response.
        - model (str): ChatGPT model to use. Default is "gpt-3.5-turbo".
    """
    import traceback
    chatbot = Chatbot(text,
        system_role=system_role, model=model, temperature=temperature, n_choices=n_choices,
        max_tokens=max_tokens
        )
    prompt = chatbot.create_prompt(task)
    firstline_pattern = r'\s?(\S*)(\n*)(.+)'
    title = re.match(firstline_pattern, text)[0]
    qna = dict()
    qna['article_title'] = title
    qna['system_role'] = system_role
    qna['prompt'] = task
    qna['model'] = model
    try:
        response = chatbot.gpt(prompt)
    except Exception as error:
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        print("An error occurred on line", lineno, "in", filename, ":", error)
        print('\t**API request failed**')
        return qna, chatbot
    try:
        for index, choice in enumerate(response.choices):
            qna[f'response_{"{:02d}".format(index+1)}'] = choice["message"]["content"]
        qna[f'text'] = text
        return qna, chatbot
    except:
        print('\t**Error with response parsing**')
        return qna, response


def batch_reply(text, prompts_df, qna_dict,  chatbot_dict, iteration_id, prompt_column='prompt',
    save_outputs=False, append_version=True,
    csv_path=r'C:\Users\silvh\OneDrive\lighthouse\Ginkgo coding\content-summarization\output',
    pickle_path=r'C:\Users\silvh\OneDrive\lighthouse\Ginkgo coding\content-summarization\output\pickles'
    ):
    """
    Send multiple summarization prompts to ChatGPT for the same text.
    Parameters:
        - prompts_df: DataFrame containing the prompts.
        - qna_dict: Dictionary to store the input and outputs.
        - iteration_id (int, float, or string): Unique ID serving as the key for results in the qna_dict
        - prompt_column (str): Name of the column in the prompts_df containing the user input.

    """
    qna_dict[iteration_id] = dict()
    prompts_df = prompts_df.reset_index(drop=True)

    for index, input in prompts_df['prompt'].items():
        # print(input)
        try:
            # prompt_num = index+1
            print(f'**Prompt #{index} of {prompts_df.index.max()}**')
            qna_dict[iteration_id][index], chatbot_dict[f'{iteration_id}_prompt{index}'] = reply(input, text)
        except:
            print('Error making chatbot request')
            break
    try:
        updated_qna_dict = pd.concat([
            pd.DataFrame(qna_dict[iteration_id]),
            prompts_df.transpose()
        ])
        qna_dict[iteration_id] = updated_qna_dict
        if save_outputs:
            try:
                filename = f"prompt_experiments_{qna_dict[iteration_id].loc['article_title', 1]}"
                savepickle(updated_qna_dict, filename=filename, path=pickle_path, append_version=append_version)
                save_csv(updated_qna_dict, filename=filename, path=pickle_path, append_version=append_version)
            except:
                print('Unable to save outputs')
        # return updated_qna_dict, chatbot_dict
    except:
        print('Error concatenating DataFrames')
    return qna_dict, chatbot_dict

def batch_summarize(text_dict, prompts_df, qna_dict,  chatbot_dict, iteration_id, prompt_column='prompt',
    save_outputs=False, filename=None, append_version=False,
    csv_path=r'C:\Users\silvh\OneDrive\lighthouse\Ginkgo coding\content-summarization\output',
    pickle_path=r'C:\Users\silvh\OneDrive\lighthouse\Ginkgo coding\content-summarization\output\pickles'
    ):
    """
    Summarize multiple texts using the same prompts.
    Parameters:
        - prompts_df: DataFrame containing the prompts.
        - qna_dict: Dictionary to store the input and outputs.
        - iteration_id (int, float, or string): Unique ID serving as the key for results in the qna_dict
        - prompt_column (str): Name of the column in the prompts_df containing the user input.

    """
    temp_qna_dict = dict()
    prompts_df = prompts_df.reset_index(drop=True)
    qna_dfs_list = []
    for key in text_dict:
        text = text_dict[key]
        temp_qna_dict[key] = dict()
        for index, input in prompts_df['prompt'].items():
            print(f'**Text #{key} prompt #{index} of {prompts_df.index.max()}**')
            try:
                temp_qna_dict[key][index], chatbot_dict[f'{iteration_id}_text{key}_prompt{index}'] = reply(input, text)
                print('\t...Success!')
            except:
                print('\t...Error making chatbot request')
                break
        try:
            updated_qna_dict = pd.concat([
                pd.DataFrame(temp_qna_dict[key]),
                prompts_df.transpose()
            ])
        except:
            print('Error concatenating prompts DataFrame')
            return temp_qna_dict, chatbot_dict
        try:
            qna_dfs_list.append(updated_qna_dict)
            qna_dict[iteration_id] = pd.concat(qna_dfs_list, axis=1)
            if save_outputs:
                try:
                    append_version = append_version if filename else False
                    filename = filename if filename else f'{datetime.now().strftime("%Y-%m-%d_%H%M")}_prompt_experiments_{"{:02d}".format(iteration_id)}'
                    savepickle(qna_dict[iteration_id], filename=filename, path=pickle_path, append_version=append_version)
                    save_csv(qna_dict[iteration_id], filename=filename, path=pickle_path, append_version=append_version)
                except:
                    print('Unable to save outputs')
        except:
            print('Error concatenating DataFrames')
    return qna_dict, chatbot_dict