import os
import sys
import json
import openai
import time
import functools
import re

from argparse import ArgumentParser

from get_gpt_response import GPTResponseGetter

sys.path.append('../')
from type.no_fake_timelines import NoFakeTimeline, TimelineData

# Define the retry decorator
def retry_decorator(max_error_count=10, retry_delay=1):
    def decorator_retry(func):
        functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_count = 0
            while error_count < max_error_count:
                try:
                    v = func(*args, **kwargs)
                    return v
                except openai.error.Timeout as e:
                    print(f"Timeout error occurred: {e}. Re-running the function.")
                    error_count += 1
                except openai.error.APIError as e:
                    print(f"OPENAI API error occurred: {e}. Re-running the function.")
                    error_count += 1
                except ValueError as e:
                    print(f"ValueError occurred: {e}. Re-running the function.")
                    error_count += 1
                except AttributeError as e: # For when other functions are called in function calling.
                    print(f"AttributeError occurred: {e}. Re-running the function.")
                    error_count += 1
                except openai.error.InvalidRequestError as e:
                    print(f"InvalidRequestError occurred: {e}. Continuing with next iteration.")
                    break
                time.sleep(retry_delay)  # If an error occurred, wait before retrying
            if error_count == max_error_count:
                sys.exit("Exceeded the maximum number of retries. Exiting the function.")
                return None
        return wrapper
    return decorator_retry

class FakeNewsSetter:
    def __init__(self):
        self.choices = ['none', 'rep0', 'rep1','rep2','rep3', 'ins0', 'ins1', 'ins2']
        '''description
        - Strings in the first half: ['rep', 'ins']
            - 'rep': means repracing method.
            - 'ins': means inserting method.
        - Numbers in the second half: ['0', '1', '2', '3']
            - '0': means no condition about contradictions.
            - '1': means don't contradict earlier documents but contradict a later document.
            - '2': means contradict any one document.
            - '3': means contradict the original document before replacement (only replacing).
        - Others
            - 'none': means no setting.
        '''

    @classmethod
    def get_choices(cls) -> list:
        instance = cls()
        return instance.choices

    @classmethod
    def decode(cls, setting: str) -> tuple:
        instance = cls()
        if not setting in instance.choices:
            sys.exit('This setting does not exist.')
        elif setting == 'none':
            sys.exit('NO setting.')
        else:
            return setting.rstrip('0123456789'), setting.lstrip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') # ex: ('rep', '0')

class FakeNewsGenerater(GPTResponseGetter):
    def __init__(self, no_fake_timelines: NoFakeTimeline, setting: str) -> None:
        self.no_fake_timelines = no_fake_timelines
        self.data = no_fake_timelines['data'][236:237]
        self.m = len(self.data)
        self.setting = setting
        self.setting_str, self.setting_num = FakeNewsSetter.decode(setting)

    def get_prompts(self, entity_items: list[str], timeline_data: TimelineData):
        docs_num, timeline = timeline_data['docs_num'], timeline_data['timeline']
        timeline = sorted(timeline, key=lambda x: x['date'])

        '''
        system content
        '''
        system_content = "You are a logical writer. You must execute the function calling's format_fake_news function to format your outputs."

        '''
        user content
        '''
        user_content = (
            "# INSTRUCTIONS\n"
            f"Below are {docs_num} documents about {entity_items}. Each document contains time information (YYYY-MM-DD), forming a timeline.\n"
            "Generate ONE fake news based on the following constraints and input documents.\n"
        )

        user_content += "# INPUT DOCUMENTS\n"
        for i, doc in enumerate(timeline):
            user_content += (
                f"document ID. {doc['ID']}\n"
                f"headline: {doc['headline']}\n"
                f"short_description: {doc['short_description']}\n"
                f"date: {doc['date']}\n"
                f"content: {doc['content']}\n"
            )


        user_content += (
            "# CONSTRAINTS\n"
            "- It needs to contain headline, short_description, date (YYYY-MM-DD), and content properties.\n"
            "- In a step-by-step manner, first generate the content and date of fake news, and then generate the headline and short description."
            "Additionally, explain why you generate such fake news and which parts of the fake news meet the following constraints"
            f"- The date of the fake news must be within a period that is later than the oldest date among the {docs_num} documents and earlier than the newest date.\n"
        )
        if self.setting_str == 'rep':
            user_content += (
                f"- Please generate fake news by replacing a suitable one among the {docs_num} documents included in the timeline I have entered.\n"
                f"- Please generat the document id and headline of the document to be replaced as remarks."
            )
        elif self.setting_str == 'ins':
            user_content += f"- Please generate fake news to be inserted in the most suitable location in the timeline I have entered.\n"
        else:
            sys.exit("Setting Error! You can choose 'rep' or 'ins'.")

        if self.setting_num == '0':
            pass
        elif self.setting_num == '1':
            user_content += (
                # "- The documents chronologically BEFORE the date of the fake news does not contradict and connects smoothly.\n"
                # "- The documents chronologically AFTER the date of the fake news clearly contradicts.\n"
                "- The fake news you generate should not contradict with ealier documents and should connect smmothly and logically.\n"
                "- However, the fake news you generate should clearly contradict the later documents.\n"
            )
        elif self.setting_num == '2':
            user_content += f"- The fake news you generate should clearly contradict with any documents in the timeline.\n"
        elif self.setting_num == '3':
            user_content += f"- The fake news you generate should clearly contradict the original document before replacement.\n"
        else:
            sys.exit("Setting Error! You can choose '0' or '1' or '2' or '3'.")

        return system_content, user_content

    def set_gpt_for_fake_news(self, model_name, temp):
        self.__model_name = model_name
        self.__temp = temp

    @retry_decorator(max_error_count=10, retry_delay=1)
    def get_fake_news(self, entity_items: list[str], timeline_data: TimelineData):
        system_content, user_content = self.get_prompts(entity_items, timeline_data)
        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content}
        ]
        # Generate fake news
        fake_news, remarks = self.get_gpt_response_fake_news(messages, self.__model_name, self.__temp)

        return fake_news, remarks

    def generate_fake_news_timelines(self):
        for i, entity_info in enumerate(self.data):
            entity_id = entity_info['entity_ID']
            entity_items = entity_info['entity_items']
            timelines = entity_info['timeline_info']['data'][:1]
            timeline_num = len(timelines)

            print(f"=== {i+1}/{self.m}. entity: {entity_items} START ===")
            for j, timeline_data in enumerate(timelines):
                print(f"=== {j+1}/{timeline_num}. fake news generating... ===")
                new_timeline = []

                for doc in timeline_data['timeline']:
                    new_doc = {
                        'ID': doc['ID'],
                        'is_fake': doc['is_fake'],
                        'headline': doc['headline'],
                        'short_description': doc['short_description'],
                        'date': doc['date'],
                        # 'content': doc['content']
                    }
                    new_timeline.append(new_doc)

                if self.setting_str == 'rep':
                    for cnt in range(10):
                        fake_news, remarks = self.get_fake_news(entity_items, timeline_data)
                        if remarks != None and self.is_valid_date_format(fake_news['date']):
                            break
                    else:
                        sys.exit('fake news generation error!')
                    print(remarks)
                    replaced_document_id = remarks['document_id']
                    new_timeline = list(filter(lambda doc: doc['ID'] != replaced_document_id, new_timeline))

                elif self.setting_str == 'ins':
                    for cnt in range(10):
                        fake_news, remarks = self.get_fake_news(entity_items, timeline_data)
                        if remarks == None:
                            break
                    else:
                        sys.exit('fake news generation error!')


                new_timeline.append(fake_news)
                new_timeline = sorted(new_timeline, key=lambda doc: doc['date'])

                new_timeline_info = {
                    'entity_id': entity_id,
                    'entity_items': entity_items,
                    'setting': self.setting,
                    'timeline': new_timeline
                }

                # ======For Test=====
                filename_test = 'fake_news_test_for_slides'
                try:
                    with open(f'/mnt/mint/hara/datasets/news_category_dataset/clustering/v1/{filename_test}.json', 'r', encoding='utf-8') as F:
                        data = json.load(F)
                except FileNotFoundError:
                    data = {
                        'name': 'Timeline Dataset with fake news.',
                        'data': []
                    }
                
                data['data'].append(new_timeline_info)

                with open(f'/mnt/mint/hara/datasets/news_category_dataset/clustering/v1/{filename_test}.json', 'w', encoding='utf-8') as F:
                    json.dump(data, F, indent=4, ensure_ascii=False, separators=(',', ': '))
                    print(f'{filename_test} is saved to {filename_test}.json')
                # ===================

                print(f"=== {j+1}/{timeline_num}. fake news DONE ===")

    def is_valid_date_format(self, date_string):
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        match = re.match(pattern, date_string)
        return bool(match)

    def set_file_to_save(self, json_file_name, out_dir):
        self.__json_file_name = json_file_name
        self.__out_dir = out_dir

    def save_fake_news_timelines(self, timeline):
        pass



def main():
    parser = ArgumentParser()
    parser.add_argument('--file_path', default='/mnt/mint/hara/datasets/news_category_dataset/clustering/v1/no_fake_timelines.json')
    parser.add_argument('--out_dir', default='/mnt/mint/hara/datasets/news_category_dataset/clustering/v1/')
    parser.add_argument('--model_name', default='gpt-4')
    parser.add_argument('--temp', default=0.8, type=float)
    parser.add_argument('--json_file_name', default='fake_news_russia_ukraine')
    parser.add_argument('--setting', default='none', choices=FakeNewsSetter.get_choices())
    args = parser.parse_args()

    with open(args.file_path, 'r') as F:
        no_fake_timelines: NoFakeTimeline = json.load(F)

    fng = FakeNewsGenerater(no_fake_timelines, args.setting)
    fng.set_gpt_for_fake_news(args.model_name, args.temp)
    fng.generate_fake_news_timelines()


'''
テストの時は
- [:]の中を確認
- get_gpt_response.pyのcontentを確認
'''

if __name__ =='__main__':
    main()