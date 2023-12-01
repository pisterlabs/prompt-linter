import copy
from typing import Dict, List, Tuple

import openai
import pandas as pd

from db_vector import db_search
from wb_chatbot import get_training_data_final
from wb_embedding import find_relevant_topics

CHAT_COMPLETIONS_MODEL = "gpt-3.5-turbo"


def generate_question_message_db(input_message: Dict[str, str],
                                 top_n: int = 1,
                                 distance_threshold: float = 0.8) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    based on a raw user question, query to add context and create a new user message
    :param distance_threshold:
    :param input_message:
    :param top_n:
    :return:
    """
    result_data = db_search(input_message['content'], top_n)
    result = result_data.loc[result_data['distance'] < distance_threshold]
    if len(result) > 0:
        new_user_message = {'role': 'user',
                            'content':
                                """context: {}
                                  Q:{}
                                  A:""".format('\n'.join(result['source']), input_message['content'])
                            }
    else:
        new_user_message = input_message
    return new_user_message, result


def generate_question_message(input_message: Dict[str, str],
                              top_n: int = 1,
                              similarity_score_threshold: float = 0.8) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    based on a raw user question, query to add context and create a new user message
    :param similarity_score_threshold:
    :param input_message:
    :param top_n:
    :return:
    """
    input_data = get_training_data_final()
    result = find_relevant_topics(input_message['content'], input_data, top_n=top_n)
    result = result.loc[result['similarity_score'] > similarity_score_threshold]
    if len(result) > 0:
        new_user_message = {'role': 'user',
                            'content':
                                """context: {}
                                  Q:{}
                                  A:""".format('\n'.join(result['source']), input_message['content'])
                            }
    else:
        new_user_message = input_message
    return new_user_message, result


def chat_message_list(message_list: List[Dict[str, str]], temperature=0.1, **kwargs: Dict) -> Dict[str, str]:
    result_msg = {}
    try:
        completion = openai.ChatCompletion.create(
            model=CHAT_COMPLETIONS_MODEL,
            messages=message_list,
            temperature=temperature,
            **kwargs
        )
        if completion is not None and len(completion['choices']) > 0:
            result_msg = completion['choices'][0]['message'].to_dict()
    except Exception as e:
        result_msg = {'role': 'assistant', 'content': f'Request failed with exception {e}'}
    return result_msg


class WBChatBot:
    def __init__(self, initial_prompt: str = None):
        self.chat_history = []
        if initial_prompt is None:
            self.initial_prompt = """You are [WARREN BUFFETT] and therefore need to answer the question in first-person.
            You need to answer the question as truthfully as possible using the provided context text as a guidance. 
            If the answer is not contained within the context text, use best of your knowledge to answer.
            If you are having problem coming up with answers, say "I don't know".
            Provide longer explanation whenever possible.
            """
        else:
            self.initial_prompt = initial_prompt
        self.chat_history.append({'role': 'system', 'content': self.initial_prompt})

    def chat(self,
             msg_question: str,
             top_n: int = 1,
             distance_threshold: float = 0.8):
        """
        create a single question function which takes the chat history into consideration
        :param distance_threshold:
        :param msg_question:
        :param top_n:
        :return:
        Usage:
        >>> self = WBChatBot()
        >>> msg_question = 'what do you think about bank failure'
        >>> self.chat(msg_question)

        >>> msg_question = "what do you think about investing in tech companies"
        >>> top_n = 1
        >>> summarize_context = False
        >>> self.chat(msg_question)

        >>> msg_question = "but you have invested in some tech companies didn't you?"
        >>> self.chat(msg_question)

        >>> msg_question = "i see, are there specific attributes in a tech company that may make you change your mind?"
        >>> self.chat(msg_question)

        >>> msg_question = "even if you might miss out alot of potential alphas for berkshire hathway?"
        >>> self.chat(msg_question)
        """
        orig_user_msg = {
            'role': 'user',
            'content': msg_question
        }
        gen_user_msg, ref_df = generate_question_message_db(orig_user_msg, top_n=top_n,
                                                            distance_threshold=distance_threshold)

        msg_list = copy.copy(self.chat_history)
        msg_list.append(gen_user_msg)
        result_msg = chat_message_list(self.chat_history)

        # we only add the original question msg to the chat history without the context,
        # this is to avoid the max token issues with openai api
        self.chat_history.append(orig_user_msg)
        self.chat_history.append(result_msg)

        return result_msg

    def chat_all(self,
                 message_list: List[Dict[str, str]],
                 top_n: int = 1,
                 distance_threshold: float = 0.8):
        """
        pass in all the messages
        :param distance_threshold:
        :param top_n:
        :param message_list:
        usage:
        >>> message_list = [{'role': 'user','content':'what do you think about tech companies'}]

        >>> message_list = [{'role': 'user','content':'what do you think about tech companies'},
        {'role': 'system','content':'i would like to invest in companies I know'},
        {'role': 'user','content':'but you have invested in tech companies in the past correct?'},
        ]

        >>> self = WBChatBot()
        >>> self.chat_all(message_list)
        """
        self.chat_reset()
        new_msg_list = copy.copy(self.chat_history)
        new_msg_list.extend(message_list)
        new_q = new_msg_list.pop()

        gen_user_msg, ref_df = generate_question_message_db(new_q, top_n=top_n,
                                                            distance_threshold=distance_threshold)

        new_msg_list.append(gen_user_msg)
        result_msg = chat_message_list(new_msg_list)
        if ref_df is not None and len(ref_df) > 0:
            result_msg['reference'] = '\n'.join(ref_df['source'])
        return result_msg

    def chat_reset(self):
        self.chat_history = []
        self.chat_history.append({'role': 'system', 'content': self.initial_prompt})
