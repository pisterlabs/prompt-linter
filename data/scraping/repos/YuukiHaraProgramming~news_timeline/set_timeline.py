import functools
import openai
import time
import sys
from typing import Tuple
from sumeval.metrics.rouge import RougeCalculator
import numpy as np
import copy

from get_gpt_response import GPTResponseGetter

sys.path.append('../')
from type.entities import EntityData, DocData
from type.no_fake_timelines import TimelineData, EntityTimelineData, Doc

# Define the retry decorator
def retry_decorator(max_error_count=10, retry_delay=1): # Loop with a maximum of 10 attempts
    def decorator_retry(func):
        functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize an error count
            error_count = 0
            while error_count < max_error_count:
                try:
                    v = func(*args, **kwargs)
                    return v
                except openai.error.Timeout as e:
                    print("Timeout error occurred. Re-running the function.")
                    print(f"Timeout error: {e}")
                    error_count += 1
                except openai.error.APIError as e:
                    print("OPENAI API error occurred. Re-running the function.")
                    print(f"OPENAI API error: {e}")
                    error_count += 1
                except ValueError as e:
                    print("ValueError occurred. Re-running the function.")
                    print(f"ValueError: {e}")
                    error_count += 1
                except TypeError as e: # For when other functions are called in function calling.
                    print("TypeError occurred. Re-running the function.")
                    print(f"TypeError: {e}")
                    error_count += 1
                except openai.error.InvalidRequestError as e:
                    print("InvalidRequestError occurred. Continuing the program.")
                    print(f"openai.error.InvalidRequestError: {e}")
                    break  # Exit the loop
                except Exception as e:
                    print("Exception error occurred. Re-running the function.")
                    print(f"Exeption: {e}")
                    error_count += 1
                time.sleep(retry_delay)  # If an error occurred, wait before retrying
            if error_count == max_error_count:
                sys.exit("Exceeded the maximum number of retries. Exiting the function.")
                return None
        return wrapper
    return decorator_retry

class TimelineSetter(GPTResponseGetter):
    '''
    docs_num_in_1timeline: The number of documents contained in ONE timeline,
    top_tl: Number of timelines to be generated, relative to the number of timelines that can be generated.
    '''
    def __init__(self, model_name, temp, min_docs_num_in_1timeline=8, max_docs_num_in_1timeline=10, top_tl=0.5):
        # super
        # super.__init__()
        # parameters
        self.model_name = model_name
        self.temp = temp
        self.min_docs_num_in_1timeline = min_docs_num_in_1timeline
        self.max_docs_num_in_1timeline = max_docs_num_in_1timeline
        self.top_tl = top_tl
        # preprocess of docs_num_in_1timeline
        self.list_docs_num_in_1timeline = list(range(self.min_docs_num_in_1timeline, self.max_docs_num_in_1timeline + 1))
        self.str_docs_num_in_1timeline = ' or '.join([str(n) for n in self.list_docs_num_in_1timeline])
        # For analytics
        self.initialize_list_for_analytics()

        '''structure of entity_info
        {
            "freq": xxx,
            "items": ["xxx", ...],
            "ID": 0,
            "docs_info": {
                "IDs": [],
                "docs": [
                    {
                        "link": xxx,
                        "headline": xxx,
                        "category": xxx,
                        "short_description": xxx,
                        "authors": xxx,
                        "date": xxx,
                        "year": xxx,
                        "month": xxx,
                        "day": xxx,
                        "content": xxx,
                        "ID": xxx,
                        "preprocessed_tokens": [
                            "xxx",
                            ...
                        ],
                        "entities_info": {
                            "num": xxx,
                            "IDs": [],
                            "entities": []
                        }
                    },
                    ...
                ]
            }
        },
        '''

    # Setting the prompts
    def get_prompts(self, entity_info: EntityData):
        '''
        system content
        '''
        system_content = "You are a professional story writer. Execute the function calling's format_timeline function."

        '''
        user content 1
        '''
        user_content_1 = (
            "# INSTRUCTIONS\n"
            "Generate the best story based on the following constraints and input statements.\n"

            "# CONSTRAINTS\n"
            f"- Generate stories based on the {entity_info['freq']} documents (consisting of headline and short description) in the INPUT STATEMENTS below.\n"
            f"- The story must be about {', '.join(entity_info['items'])}.\n"
            "- It must be less than 100 words.\n"

            "# INPUT STATEMENTS\n"
            # "Document ID. headline: short description"
        )
        for doc in entity_info['docs_info']['docs']:
            doc_ID, headline, short_description = doc['ID'], doc['headline'], doc['short_description']
            user_content_1 += f"Document ID -> {doc_ID}, document -> {headline}: {short_description}\n"
            # user_content_1 += f"{doc_ID}. {headline}: {short_description}\n"
        user_content_1 += "# OUTPUT STORY\n"

        '''
        user content 2
        '''
        user_content_2 = (
            "# INSTRUCTIONS\n"
            f"Select \" at least {self.min_docs_num_in_1timeline}\", and \"at most {self.max_docs_num_in_1timeline}\" documents that are most relevant to the above story.\n"
            # f"Pick \"{self.str_docs_num_in_1timeline}\" documents that are most relevant to the above story.\n"
            # "When responding, generate the Document ID, the headline: short_description of the document, and which sentence in the story the document is most related to.\n"
            # "The headline and short_description you choose must COVER most of the story.\n"
            "Most of the story you generated MUST BE COVERED by the headline and short_description you choose.\n"
            "Please follow these three conditions and generate by using the following OUTPUT FORMAT.\n"
            # f"When responding for \"{self.str_docs_num_in_1timeline}\" documents, please follow these two conditions and generate by using the following OUTPUT FORMAT \"{self.str_docs_num_in_1timeline}\" times.\n"

            "# CONDITIONS\n"
            f"1. First of all, please generate how many documents you have picked out of {self.str_docs_num_in_1timeline}.\n"
            "2. generate the Document ID, the headline: short_description of the document.\n"
            "3. generate REASONS why you chose this document and clearly point out the STATEMENT in the story you generated which is most relevant to this document.\n"

            "# OUTPUT FORMAT\n"
            "1. Number of documents -> \n"
            "2. ID -> , document -> \n"
            "3. REASONS and STATEMENT -> \n"
        )

        return system_content, user_content_1, user_content_2

    def get_prompts_for_rouge(self, story: str, entity_info: EntityData):
        '''
        system content
        '''
        system_content = "You are a good reader and logical thinker. Execute the function calling's format_timeline function."

        '''
        user content
        '''
        user_content = (
            "# INSTRUCTIONS\n"
            "Select only ONE document that are most relevant to the below STORY, from the below INPUT DOCUMENTS.\n"
            "Explain ONE reason why you chose that document.\n"
            "Please generate by using the following OUTPUT FORMAT.\n"

            "# STORY\n"
            f"{story}"

            "# OUTPUT FORMAT\n"
            "1. Number of documents -> 1\n"
            "2. ID -> , document -> \n"
            "3. REASONS and STATEMENT -> \n"

            "# INPUT DOCUMENTS\n"
        )
        for doc in entity_info['docs_info']['docs']:
            doc_ID, headline, short_description = doc['ID'], doc['headline'], doc['short_description']
            user_content += f"Document ID -> {doc_ID}, documents -> {headline}: {short_description}\n"

        return system_content, user_content

    def sort_docs_by_date(self, docs: list[Doc], ascending: bool = True) -> list[Doc]:
        # Sort the list of docs using the 'date' key.
        # The order is ascending by default; if ascending=False, the list will be sorted in descending order.
        return sorted(docs, key=lambda x: x['date'], reverse=not ascending)

    def _delete_dicts_by_id(self, dictionary_list: list[dict], id_list: list[int]) -> list[dict]:
        # Find the indexes of dictionaries with IDs to be removed
        indexes_to_remove = [i for i, item in enumerate(dictionary_list) if item['ID'] in id_list]
        # Create a new list without the dictionaries at the identified indexes
        filtered_list = [item for i, item in enumerate(dictionary_list) if i not in indexes_to_remove]
        return filtered_list

    def _check_conditions(self, entity_info: EntityData, IDs_from_gpt: list[int], timeline_list: list[Doc]) -> bool:
        # Check number of docs
        cond1 = len(IDs_from_gpt) in self.list_docs_num_in_1timeline
        # Check document ID
        cond2 = set(IDs_from_gpt) <= set(entity_info['docs_info']['IDs'])
        # Check the length between IDs_from_gpt and timeline_list
        cond3 = len(timeline_list) == len(IDs_from_gpt)
        return cond1 and cond2 and cond3

    """
    Coverage by rouge score
    """
    def set_rouge_parms(self, alpha=0.8, th_1=0.25, th_2=0.12, th_l=0.15, rouge_used=True):
        self.rouge_used = rouge_used
        self.rouge_alpha = alpha if rouge_used else None
        self.rouge_th_1 = th_1 if rouge_used else None
        self.rouge_th_2 = th_2 if rouge_used else None
        self.rouge_th_l = th_l if rouge_used else None

    def get_rouge_scores(self, summary: str, reference: str, alpha: float):
        rouge = RougeCalculator(stopwords=True, lang="en")
        rouge_1 = rouge.rouge_n(summary=summary, references=reference, n=1, alpha=alpha)
        rouge_2 = rouge.rouge_n(summary=summary, references=reference, n=2, alpha=alpha)
        rouge_l = rouge.rouge_l(summary=summary, references=reference, alpha=alpha)
        scores = {
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l
        }
        return scores

    def _check_coverage_by_rouge(self, timeline_list: list[Doc], story: str) -> bool:
        docs_list = []
        for doc in timeline_list:
            docs_list.append(f"{doc['headline']} {doc['short_description']}")
        docs_str = " ".join(docs_list)

        # Calcurate rouge score
        scores = self.get_rouge_scores(summary=story, reference=docs_str, alpha=self.rouge_alpha)
        rouge_1 = scores['rouge_1']
        rouge_2 = scores['rouge_2']
        rouge_l = scores['rouge_l']

        THRESHOLD = {
            # 'rouge_1': 0.25,
            # 'rouge_2': 0.12,
            # 'rouge_l': 0.15,
            'rouge_1': self.rouge_th_1,
            'rouge_2': self.rouge_th_2,
            'rouge_l': self.rouge_th_l,
        }
        boolean: bool = rouge_1 > THRESHOLD['rouge_1'] or rouge_2 > THRESHOLD['rouge_2'] or rouge_l > THRESHOLD['rouge_l']

        # print(f"ROUGE-1: {rouge_1}, ROUGE-2: {rouge_2}, ROUGE-L: {rouge_l}")
        # if not boolean:
        #     print('Coverage by rouge score is not enough.')
        #     print(f"Timeline num: {len(timeline_list)}")
        return boolean

    @retry_decorator(max_error_count=10, retry_delay=1)
    def generate_timeline_by_rouge(self, story: str, timeline_list: list[Doc], ID_list: list[int], entity_info: EntityData, useGPT=True) -> Tuple[list[Doc], list[int], bool]:
        RE_GENERATE_FLAG = False
        # print(f"docs num: {len(entity_info['docs_info']['IDs'])}")
        if self._check_coverage_by_rouge(timeline_list, story) and len(timeline_list) >= self.min_docs_num_in_1timeline:
            return timeline_list, ID_list, RE_GENERATE_FLAG
        elif len(timeline_list) >= self.max_docs_num_in_1timeline:
            return timeline_list, ID_list, not RE_GENERATE_FLAG
        else:
            # Update entity info left
            entity_info['docs_info'] = {
                'IDs': list(set(entity_info['docs_info']['IDs']) - set(ID_list)),
                'docs': self._delete_dicts_by_id(entity_info['docs_info']['docs'], ID_list)
            }
            entity_info['freq'] = len(entity_info['docs_info']['IDs'])

            # Define
            new_doc: Doc
            new_id: int

            """
            Select ONE document
            useGPT True -> select a document by GPT
            useGPT False -> select a document by max rouge_1 score
            """
            if useGPT:
                system_content, user_content = self.get_prompts_for_rouge(story, entity_info)
                # messages
                messages=[
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': user_content}
                ]

                # Loop processing
                for i in range(32):
                    new_doc_tuple = self.get_gpt_response_timeline(messages, model_name=self.model_name, temp=0)
                    if len(new_doc_tuple[0]) == len(new_doc_tuple[1]) == 1:
                        # Get a new document and its ID
                        new_doc = new_doc_tuple[0][0]
                        new_id = new_doc_tuple[1][0]
                        break
                    else:
                        print(f"Re-choice for the {i + 1}-th time (generate_timeline_by_rouge in set_timeline.py)")
                else:
                    raise Exception("Couldn't choose a new document. (generate_timeline_by_rouge in set_timeline.py)")
            else:
                rouge_list = []
                id_list = []
                timeline_docs_list: list[str] = [f"{doc['headline']} {doc['short_description']}" for doc in timeline_list]
                # for doc in timeline_list:
                #     timeline_docs_list.append(f"{doc['headline']} {doc['short_description']}")
                for doc_data in entity_info['docs_info']['docs']:
                    doc_data: DocData
                    docs_list = [f"{doc_data['headline']} {doc_data['short_description']}"]
                    docs_list.extend(timeline_docs_list)
                    docs_str = " ".join(docs_list)

                    scores = self.get_rouge_scores(summary=story, reference=docs_str, alpha=0.8)
                    rouge_list.append(scores['rouge_1'])
                    id_list.append(doc_data['ID'])

                # max_id = id_list[np.argmax(np.array(rouge_list))]
                try:
                    max_id = id_list[np.argmax(np.array(rouge_list))]
                except ValueError as e:
                    print(f"ValueError here. {e}")
                    print(f"rouge_list: {rouge_list}")
                    sys.exit("STOP")
                for doc_data in entity_info['docs_info']['docs']:
                    if doc_data['ID'] == max_id:
                        new_id = max_id
                        new_doc = {
                            'ID': doc_data['ID'],
                            'is_fake': False,
                            'document': f"{doc_data['headline']}: {doc_data['short_description']}",
                            'headline': doc_data['headline'],
                            'short_description': doc_data['short_description'],
                            'date': doc_data['date'],
                            'content': doc_data['content'],
                            'reason': "No reasons bacause of no-GPT."
                        }
                        break

            timeline_list.append(new_doc)
            ID_list.append(new_id)

            # For timeline_info_archive
            self.append_timeline_info_archive({'max_score': max(rouge_list), 'timeline_list': timeline_list, 'ID_list': ID_list})

            # Execute reccurently
            return self.generate_timeline_by_rouge(story, timeline_list, ID_list, entity_info, useGPT)


    """
    Define other Setter and Getter
    """

    def initialize_list_for_analytics(self):
        self.no_timeline_entity_id = []
        self.analytics_docs_num = []
        self.analytics_reexe_num = []

    def get_max_reexe_num(self) -> int:
        return self.__max_reexe_num

    def set_max_reexe_num(self, value: int) -> None:
        self.__max_reexe_num = value

    def set_reexe_num(self, value: int) -> None:
        self.__reexe_num = value

    def set_timeline_info_archive(self, value: list) -> None:
        self.__timeline_info_archive = value

    def get_timeline_info_archive(self) -> list:
        return self.__timeline_info_archive

    def initialize_timeline_info_archive(self) -> None:
        self.set_timeline_info_archive([])

    def append_timeline_info_archive(self, value: dict) -> None:
        self.__timeline_info_archive.append(value)

    """
    (1/3) Generate a timeline without rouge score
    """
    @retry_decorator(max_error_count=10, retry_delay=1)
    def generate_story_and_timeline_without_rouge(self, entity_info: EntityData) -> Tuple[TimelineData, list[int]]:
        # setter for GPTResponseGetter
        self.set_entity_info(entity_info)

        # prompts
        system_content, user_content_1, user_content_2 = self.get_prompts(entity_info)

        # Loop processing
        for i in range(self.__max_reexe_num+1):
            self.set_reexe_num(i)

            # messages
            messages=[
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content_1}
            ]

            messages = self.get_gpt_response_classic(messages, model_name=self.model_name, temp=self.temp)
            story: str = messages[-1]['content']
            print('Got 1st GPT response.')

            messages.append({'role': 'user', 'content': user_content_2})
            timeline_list, IDs_from_gpt = self.get_gpt_response_timeline(messages, model_name=self.model_name, temp=0)
            timeline_list = self.sort_docs_by_date(timeline_list, True)

            #Check
            if self._check_conditions(entity_info, IDs_from_gpt, timeline_list):
            # if self._check_conditions(entity_info, IDs_from_gpt, timeline_list) and self._check_coverage_by_rouge(timeline_list, story):
                # If all conditions are met
                print('Got 2nd GPT response.')
                break # Exit the loop as conditions are met
            else:
                print('Failed to get 2nd GPT response.')
                messages = []
                if i != self.__max_reexe_num:
                    print(f"Re-execution for the {i + 1}-th time")
        else:
            print('NO TIMELINE INFO')
            self.set_reexe_num(-1)
            self.no_timeline_entity_id.append(entity_info['ID'])
            story = ''
            timeline_list = []

        # Update timelines
        timeline_info: TimelineData = {
            'reexe_num': self.__reexe_num,
            'docs_num': len(timeline_list),
            'story': story,
            'timeline': timeline_list
        }

        return timeline_info, IDs_from_gpt


    """
    (2/3) Generate a timeline with rouge score and multi-docs start
    """
    @retry_decorator(max_error_count=10, retry_delay=1)
    def generate_story_and_timeline_with_mutli_docs_rouge(self, entity_info: EntityData) -> Tuple[TimelineData, list[int]]:
        # setter for GPTResponseGetter
        self.set_entity_info(entity_info)

        # prompts
        system_content, user_content_1, user_content_2 = self.get_prompts(entity_info)

        # Loop processing
        for i in range(self.__max_reexe_num+1):
            self.set_reexe_num(i)

            # messages
            messages=[
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content_1}
            ]

            messages = self.get_gpt_response_classic(messages, model_name=self.model_name, temp=self.temp)
            story: str = messages[-1]['content']
            print('Got 1st GPT response.')

            messages.append({'role': 'user', 'content': user_content_2})
            timeline_list, IDs_from_gpt = self.get_gpt_response_timeline(messages, model_name=self.model_name, temp=0)

            #Check
            if self._check_conditions(entity_info, IDs_from_gpt, timeline_list):
                # If all conditions are met
                break # Exit the loop as conditions are met
            else:
                print('Failed to get 2nd GPT response.')
                messages = []
                if i != self.__max_reexe_num:
                    print(f"Re-execution for the {i + 1}-th time")
        else:
            print('NO TIMELINE INFO')
            self.set_reexe_num(-1)
            self.no_timeline_entity_id.append(entity_info['ID'])
            story = ''
            timeline_list = []

        # Add some docs by rouge score
        entity_info_copied = copy.deepcopy(entity_info)
        timeline_list, IDs_from_gpt, _ = self.generate_timeline_by_rouge(story, timeline_list, IDs_from_gpt, entity_info_copied, useGPT=False)
        timeline_list = self.sort_docs_by_date(timeline_list, True)
        print('Got 2nd GPT response.')

        # Update timelines
        timeline_info: TimelineData = {
            'reexe_num': self.__reexe_num,
            'docs_num': len(timeline_list),
            'story': story,
            'timeline': timeline_list
        }

        return timeline_info, IDs_from_gpt

    """
    (3/3) Generate a timeline with rouge score and single-doc start
    """
    @retry_decorator(max_error_count=10, retry_delay=1)
    def generate_story_and_timeline_with_single_doc_rouge(self, entity_info: EntityData) -> Tuple[TimelineData, list[int]]:
        # setter for GPTResponseGetter
        self.set_entity_info(entity_info)

        # prompts
        system_content, user_content_1, _ = self.get_prompts(entity_info)


        # Loop processing
        for i in range(3):
            self.set_reexe_num(i)

            # messages
            messages=[
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content_1}
            ]
            messages = self.get_gpt_response_classic(messages, model_name=self.model_name, temp=self.temp)
            story: str = messages[-1]['content']
            print('Got 1st GPT response.')

            # Add some docs by rouge score
            self.initialize_timeline_info_archive()
            entity_info_copied = copy.deepcopy(entity_info)
            new_timeline_list, new_id_list, re_generate_flag = self.generate_timeline_by_rouge(story, [], [], entity_info_copied, useGPT=False)

            if not re_generate_flag:
                print(f'Got a timeline (len = {len(new_id_list)}).')
                break
            else:
                print('Missed to create a timeline.')
        else:
            # In the case of i==2 and re_generate_flag==True
            timeline_info_archive = self.get_timeline_info_archive()
            archive_rouge_scores: list[float] = [item['max_score'] for item in timeline_info_archive]
            max_score_id: int = np.argmax(np.array(archive_rouge_scores))
            new_timeline_list = timeline_info_archive[max_score_id]['timeline_list']
            new_id_list = timeline_info_archive[max_score_id]['ID_list']
            print(f'Got a compromised timeline (len = {len(new_id_list)}).')

        timeline_list = self.sort_docs_by_date(new_timeline_list, True)
        IDs_from_gpt = new_id_list

        # Update timelines
        timeline_info: TimelineData = {
            'reexe_num': self.__reexe_num,
            'docs_num': len(timeline_list),
            'story': story,
            'timeline': timeline_list
        }

        return timeline_info, IDs_from_gpt


    """
    Generate timelines about ONE keyword group
    """
    def generate_timelines(self, entity_info_left: EntityData, timeline_num: int) -> EntityTimelineData:
        entity_ID, entity_items = entity_info_left['ID'], entity_info_left['items']
        list_to_save_timelines: list[TimelineData] = []
        docs_IDs: list[int] = []

        # Initialize for analytics list
        self.initialize_list_for_analytics()
        for i in range(timeline_num):
            print(f'=== {i+1}/{timeline_num}. START ===')

            if self.rouge_used:
                # timeline_info, IDs_from_gpt = self.generate_story_and_timeline_with_mutli_docs_rouge(entity_info_left)
                timeline_info, IDs_from_gpt = self.generate_story_and_timeline_with_single_doc_rouge(entity_info_left)
            else:
                timeline_info, IDs_from_gpt = self.generate_story_and_timeline_without_rouge(entity_info_left)

            if len(IDs_from_gpt) > 0 and timeline_info['reexe_num'] != -1:
                list_to_save_timelines.append(timeline_info)
                docs_IDs.extend(IDs_from_gpt)
                self.analytics_docs_num.append(timeline_info['docs_num'])
            self.analytics_reexe_num.append(timeline_info['reexe_num'])

            # Update entity info left
            entity_info_left['docs_info'] = {
                'IDs': list(set(entity_info_left['docs_info']['IDs']) - set(IDs_from_gpt)),
                'docs': self._delete_dicts_by_id(entity_info_left['docs_info']['docs'], IDs_from_gpt)
            }
            entity_info_left['freq'] = len(entity_info_left['docs_info']['IDs'])

            print(f'=== {i+1}/{timeline_num}. DONE ===')

        output_data: EntityTimelineData = {
            'entity_ID': entity_ID,
            'entity_items': entity_items,
            'docs_info': {
                'IDs': docs_IDs
            },
            'timeline_info': {
                'timeline_num': len(list_to_save_timelines),
                'data': list_to_save_timelines
            }
        }

        return output_data
