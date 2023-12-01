import os
import pprint
from typing import Dict, List, Tuple

import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from db_vector import insert_data_to_milvus_only
from wb_embedding import count_tokens, get_embeddding, add_token_count_to_wb_transcript, add_embedding_to_wb_transcript


def run_model():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = """
  Q: If warren buffett was 20 year old today, how would he invest his money
  A:"""

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(response['choices'][0]['text'])


def get_training_data() -> List[Dict[str, str]]:
    url = 'https://buffettfaq.com/'
    req = requests.get(url)
    req.encoding = 'UTF-8'
    soup = BeautifulSoup(req.content, 'html.parser')
    all_h3_headings = soup.find_all('h3')
    result_list = []
    for h3 in all_h3_headings:
        # find the end of the paragraph which is a link back to 'question'
        p_element = h3.next_element
        answer_text = ''
        while p_element is not None:
            print(p_element)
            if p_element.name == 'p':
                answer_text += p_element.text + ' '
            next_element = p_element.next_element
            if next_element is not None and \
                    isinstance(next_element, Tag) and \
                    'class' in next_element.attrs.keys() and \
                    next_element.attrs['class'][0] == 'source':
                break
            else:
                p_element = p_element.next_element
        result_list.append({'prompt': h3.text, 'completion': answer_text})
    return result_list


def save_training_file():
    training_data = get_training_data()
    input_df = pd.DataFrame(training_data)
    input_df.to_json('wb_train_data.json', orient='records', lines=True)


def get_training_data_final():
    output_df = pd.read_pickle('wb_train_data_full_transcript.pkl')
    return output_df


def export_data_milvus_format():
    df = get_training_data_final()
    result_json_str = df.iloc[3:].to_json(orient='records')
    result_json_str = '{ "rows":' + result_json_str + '}'
    with open("wb_train_data_full_transcript.json", "w") as outfile:
        outfile.write(result_json_str)


def get_training_data_wb_transcript(input_df: pd.DataFrame, combine_topic: bool = True):
    """

    :param input_df:
    :param combine_topic:
    :return:

    Usage:
    >>> input_df = pd.read_pickle('wb_train_data_full_transcript.pkl')
    >>> combine_topic = True
    """
    output_df = input_df
    if combine_topic:
        final_list = []
        for idx, g in input_df.groupby('category'):
            convo_list = g.apply(lambda x: '[{}]: {}'.format(x['speaker'], x['content']), axis=1).values
            title_fix = '. '.join(idx.split('. ')[1:])
            result_tuple = (title_fix, ' \n '.join(convo_list), g['source'].values[0])
            final_list.append(result_tuple)
        output_df = pd.DataFrame(final_list, columns=['category', 'content', 'source'])
    return output_df


def parse_insert(url_list: List[str]):
    insert_df = parse_wb_specific_url_list(url_list)
    insert_data_to_milvus_only(insert_df)


def parse_wb_specific_url_list(url_list: List[str]):
    """

    :param url_list:
    Usage:
    >>> url_list = ['https://buffett.cnbc.com/video/2023/05/08/morning-session---2023-meeting.html',
    'https://buffett.cnbc.com/video/2023/05/08/afternoon-session---2023-meeting.html']
    >>> parse_wb_specific_url_list(url_list)
    """
    result_list = []
    for url in url_list:
        print('process [{}]'.format(url))
        out_df = parse_cnbc_official_transcript_df(url)
        if out_df is None:
            print('skip [{}]'.format(url))
        else:
            result_list.append(out_df)
    final_df = pd.concat(result_list, ignore_index=True)
    final_df = get_training_data_wb_transcript(final_df)
    final_df = add_token_count_to_wb_transcript(final_df)
    final_df = add_embedding_to_wb_transcript(final_df)
    return final_df


def parse_wb_all():
    url_list = parse_wb_url_list()
    result_list = []
    for url in url_list:
        print('process [{}]'.format(url))
        out_df = parse_cnbc_official_transcript_df(url)
        if out_df is None:
            print('skip [{}]'.format(url))
        else:
            result_list.append(out_df)
    final_df = pd.concat(result_list, ignore_index=True)
    final_df.to_pickle('wb_train_data_full_transcript.pkl')


def parse_wb_url_list():
    base_url = 'https://buffett.cnbc.com/annual-meetings/'
    req = requests.get(base_url)
    req.encoding = 'UTF-8'
    soup = BeautifulSoup(req.content, 'html.parser')
    meeting_card_list = soup.find_all("a", {"class": "MeetingCard-meetingCard"})
    url_list = []
    for meeting in meeting_card_list:
        meeting_url = meeting.get('href')
        req_meeting = requests.get(meeting_url)
        req_meeting.encoding = 'UTF-8'
        soup_meeting = BeautifulSoup(req_meeting.content, 'html.parser')
        meeting_video_url_list = soup_meeting.find_all('a', {'class': "Card-mediaContainer"})
        for meet_video in meeting_video_url_list:
            if meet_video.find('span', {'class': 'Card-videoLabel'}).text == 'WATCH FULL VIDEO':
                url_list.append(meet_video.get('href'))
    return url_list


def parse_cnbc_official_transcript_df(input_url: str, combine_sentence_per_speaker: bool = True) -> pd.DataFrame:
    """

    :param combine_sentence_per_speaker:
    :param input_url:
    :return:

    Usage:
    >>> input_url = 'https://buffett.cnbc.com/video/2020/05/04/berkshire-hathaway-annual-meeting--may-02-2020.html'
    >>> combine_sentence_per_speaker = True
    >>> parse_cnbc_official_transcript_df(input_url, True)
    """
    data_list = parse_cnbc_official_transcript(input_url)
    df_result = None
    if len(data_list) > 0:
        df_list = []
        for data_each in data_list:
            df_temp = pd.DataFrame(data_each[1], columns=['speaker', 'content'])
            df_temp['category'] = data_each[0]
            df_list.append(df_temp)
        df_result = pd.concat(df_list, ignore_index=True)

        df_result['source'] = input_url
        new_combined_list = []
        if combine_sentence_per_speaker:
            category_g = df_result.groupby('category')
            for i_g, g in category_g:
                g['speaker_change'] = g['speaker'].shift() != g['speaker']
                result_row_list = []
                new_row = None
                for idx in range(len(g)):
                    current_row = g.iloc[idx]
                    current_row_content = current_row['content']
                    if current_row['speaker_change']:
                        if new_row is not None:
                            result_row_list.append(new_row)
                        new_row = current_row.copy().drop('speaker_change')
                    else:
                        new_row['content'] = new_row['content'] + ' ' + current_row_content
                # append the last section
                if new_row is not None:
                    result_row_list.append(new_row)
                new_category_df = pd.DataFrame(result_row_list).reset_index(drop=True)
                new_combined_list.append(new_category_df)
        df_result = pd.concat(new_combined_list, ignore_index=True)
    return df_result


def parse_cnbc_official_transcript(input_url: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """

    :param input_url:

    usage:
    >>> input_url = 'https://buffett.cnbc.com/video/2020/05/04/berkshire-hathaway-annual-meeting--may-02-2020.html'
    """
    req = requests.get(input_url)
    req.encoding = 'UTF-8'
    soup = BeautifulSoup(req.content, 'html.parser')
    chapter_list = soup.find_all("div", {"class": "Chapter-chapter"})
    result_chapter_list = []
    for chap in chapter_list:
        # first find the title of the chapter
        chap_title = chap.findChild("div", {"class": "Chapter-chapterTitle"})
        if chap_title is not None:
            chap_title_text = chap_title.text
        else:
            chap_title_text = ''
        child_speaker = chap.findChildren("div", {"class": "Chapter-chapterSpeakerWrapper"})
        # result in list of dictionary of list of string
        convo_list = []
        speaker_id = ''

        for c in child_speaker:
            speaker_text = c.findChild("p").text
            speaker_id_temp = speaker_text.split(':')[0]
            if speaker_id_temp.upper() == speaker_id_temp:
                # new speaker
                speaker_id = speaker_id_temp
                speaker_text = speaker_text[len(speaker_id_temp) + 2:]
            list_content = (speaker_id, speaker_text)
            convo_list.append(list_content)
        chap_list = (chap_title_text, convo_list)
        # chap_df = pd.DataFrame()
        result_chapter_list.append(chap_list)
    return result_chapter_list
