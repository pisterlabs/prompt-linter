import time

import openai
import pandas as pd
import ai21
import re
from Access_to_models import chatgpt

forward_flow_prompt = "Starting with the word seed_word, name the next word that follows in your mind from the previous " \
                      "word. Please put down only single words, and do not use proper nouns " \
                      "(such as names, brands, etc.). Name 19 words in total. Seperate the words by comma."

seedwords_study5_6 = ['paper', 'snow', 'table', 'candle', 'bear', 'toaster' ]


def save_response_ff(prompt, seedword, filename, num_responses, temperature):
    """
       Save responses for a given prompt and seedword to a file.

       This function generates responses using the specified prompt and seedword and saves them to a file.

       Parameters:
           prompt (str): The base prompt for generating the responses.
           seedword (str): The seedword to be used in the prompt.
           filename (str): The name of the file to save the responses.
           num_responses (int): The number of responses to generate and save.
           temperature (float): The temperature for response generation (higher values make output more random).

       Returns:
           None
    """
    prompt = prompt.replace('seed_word', seedword)
    print(prompt)
    # print(df)
    try:
        df = pd.read_csv(filename)
        lines = len(df)
        print('curr num of lines', lines)
    except FileNotFoundError:
        lines = 0
        df = pd.DataFrame(columns=['Subject#','Word 1','Word 2','Word 3','Word 4','Word 5','Word 6','Word 7','Word 8','Word 9',
                                    'Word 10','Word 11','Word 12','Word 13','Word 14','Word 15','Word 16','Word 17','Word 18',
                                   'Word 19','Word 20'])
    if lines < num_responses:
        try:
            response = chatgpt(prompt, temperature)
            # response = chatgpt2(prompt, top_p)
            print('response', response)
            response = ''.join(filter(lambda x: (not x.isdigit() and not x in ['.', ' ']), response))
            response = response.split(',')
            # response = response.split('\n')
            if len(response) == 1:
                response = "".join(response)
                response = response.split('\n')
                if len(response) == 1:
                    response = "".join(response)
                    response = response.split('->')
                    if len(response) == 1:
                        response = "".join(response)
                        response = response.split('-')
            if len(response) < 20:
                for j in range(20 - len(response)):
                    response.append('')
            df.loc[len(df.index)] = [lines] + response
            df.to_csv(filename, index=False)
            save_response_ff(prompt, seedword, filename, num_responses, temperature)
        except openai.error.RateLimitError as e:
            print(e)
            print('rate limit')
            time.sleep(10)
            save_response_ff(prompt, seedword, filename, num_responses, temperature)
    else:
        return None


def filter_responses(filename):
    """
        Filter and modify responses in the given file.

        This function reads responses from the specified file, filters out certain unwanted characters,
        and writes the modified responses back to a new file.

        Parameters:
            filename (str): The name of the file containing the responses.

        Returns:
            None
        """
    df = pd.read_csv(filename, sep=',', index_col=0)
    for i in range(1,500):
        try:
            df['Word ' + str(i)] = df['Word ' + str(i)].astype(str)
            df['Word ' + str(i)] = df['Word ' + str(i)].str.replace(r'[^a-zA-Z]', '')
            df['Word ' + str(i)] = df['Word ' + str(i)]
        except KeyError:
            pass
    print(df)
    df = df.drop(df.columns[(df == 'nan').all()], axis=1)
    for j in range(50,500):
        try:
            df = df.drop(columns = ['Word ' + str(j)])
        except KeyError:
            pass
    df.to_csv('filtered_'+filename)


def make_seed_word_first_word(filename, seedword):
    df = pd.read_csv(filename)
    # Iterate through each row and perform the shift
    for index, row in df.iterrows():
        if row['Word 1'] != seedword:
            print('true')
            for i in range(len(row) - 2, 0, -1):
                df.at[index, f'Word {i + 1}'] = row[f'Word {i}']
            df.at[index, 'Word 1'] = seedword
    pd.set_option('display.expand_frame_repr', False)
    print(df)
    df.to_csv(filename, index=False)



# example run
# save_response_ff(forward_flow_prompt, 'bear', 'ff_test.csv', 100, 1)
# make_seed_word_first_word('ff_test.csv', 'bear')
