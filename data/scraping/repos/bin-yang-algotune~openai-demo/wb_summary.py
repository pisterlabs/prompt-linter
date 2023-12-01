import math
import time

import openai
import os
import pandas as pd


def summarize_block(input_df: pd.DataFrame):
    """

    :param input_df:
    usage:
    >>> input_df = pd.read_pickle('wb_train_data_transcript_2023.pkl')
    """
    overall_text = []
    for idx, irow in input_df.iterrows():
        print('processing [{}]'.format(irow['category']))
        content = irow['content']
        response = summarize_section(content)
        time.sleep(10)
        overall_text.append(response)

    summary_list = []
    batch_size = 5
    for idx in range(int(math.ceil(len(overall_text) / batch_size))):
        batch_list = overall_text[idx*batch_size: (idx+1)*batch_size]
        batch_text = ' '.join(batch_list)
        batch_summary = summarize_section(batch_text)
        summary_list.append(batch_summary)
    overall_text_flat = ' '.join(batch_summary)
    return overall_text_flat


def summarize_section(input_text: str):
    """

    :param input_text:
    usage:
    >>> input_text = pd.read_pickle('wb_train_data_transcript_2023.pkl')['content'].values[3]
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = """
    You are [WARREN BUFFETT] and therefore need to answer the question in first-person.
    Summarize the following but do not say In summary:
    
    {}""".format(input_text)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=425,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].replace('[WARREN BUFFETT]:', '')
