import copy
import threading
import tqdm
import openai
from transformers import AutoTokenizer
import time
from typing import List, Dict, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')


def truncate_doc_in_user_prompt(user_prompt):
    last_index = user_prompt.rfind('Content:\n')
    if last_index != -1:
        user_prompt = user_prompt[:last_index]

    user_prompt = '\n'.join(user_prompt.split('\n')[:-2])
    user_prompt = user_prompt.strip()
    return user_prompt


def letter_to_int(letter):
    if 'a' <= letter <= 'z':
        return ord(letter) - ord('a')
    elif 'A' <= letter <= 'Z':
        return ord(letter) - ord('A')
    else:
        print('letter:{}'.format(letter))
        raise NotImplementedError


def letter_to_int_upper(letter):
    if 'A' <= letter <= 'Z':
        return ord(letter) - ord('A')
    else:
        print('letter:{}'.format(letter))
        raise NotImplementedError


def letter_to_int_lower(letter):
    if 'a' <= letter <= 'z':
        return ord(letter) - ord('a')
    else:
        print('letter:{}'.format(letter))
        raise NotImplementedError


def int_to_letter_lower(n):
    if 0 <= n <= 25:
        return chr(n + ord('a'))
    else:
        raise ValueError('The entered integer must be between 0 and 25')


def int_to_letter_upper(n):
    if 0 <= n <= 25:
        return chr(n + ord('A'))
    else:
        raise ValueError('The entered integer must be between 0 and 25')


def create_stage2_select_prompt(questions: List[str], 
                                docs: List, 
                                k: int, 
                                idf_use_letter: str, 
                                use_title: int, 
                                stage2_select_system_prompt: str, 
                                used_doc_field: str, 
                                reverse_doc_order: bool=False) -> List[Dict]:
    """Create the prompt for selection in the 2nd stage.
    Args
    ----
    questions: List[str]
        The question.
    docs: List
        The documents relevant to the question
    k: int
        A specified number of documents for answering the user's specific question(s).
    idf_use_letter: str
        Use uppercase letters, lowercase letters, or integers to mark the documents.
    use_title: int
        Whether to use title or not.
    stage2_select_system_prompt: str
        System prompt for instruction.
    used_doc_field_in_retrieval: str
        Which filed of document to use in retrieval.
    reverse_doc_order: bool=False
        Whether to reverse the order of document or not.

    Returns
    -------
    prompt: List[Dict]
        Prompt for selection.
    """
    user_prompt = 'Question:\n{}\n\nk: {}\n\n'.format('\n'.join(questions), k)
    user_prompt += 'Candidate Documents:\n\n'

    prompt_doc_str_list = []

    for i, doc in enumerate(docs):
        if idf_use_letter == 'lower':
            idf = int_to_letter_lower(i)
        elif idf_use_letter == 'upper':
            idf = int_to_letter_upper(i)
        else:
            idf = i + 1
        if use_title:
            prompt_doc_str_list.append('{}\nTitle:\n{}\nContent:\n{}\n\n'.format(idf, doc['title'], doc[used_doc_field]))
        else:
            prompt_doc_str_list.append('{}\nContent:\n{}\n\n'.format(idf, doc[used_doc_field]))

    if reverse_doc_order:
        user_prompt += ''.join(list(reversed(prompt_doc_str_list)))
    else:
        user_prompt += ''.join(prompt_doc_str_list)

    prompt = [
        {'role': 'system', 'content': stage2_select_system_prompt},
        {'role': 'user', 'content': user_prompt.strip()}
    ]

    return prompt


def select_k_supporting_documents(questions: List[str], 
                                  tmp_selected_docs: List, 
                                  extra_docs_to_browse: List[Dict], 
                                  k: int, 
                                  selected_doc_first: int,
                                  idf_use_letter: str, 
                                  use_title: int, 
                                  model_name: str, 
                                  stage2_select_system_prompt: str, 
                                  used_doc_field_in_retrieval: str, 
                                  thread: "instance") -> Dict:
    """Select k supporting documents.
    Args
    ----
    questions: List[str]
        The question.
    tmp_selected_docs: List
        pass
    extra_docs_to_browse: List[Dict]
        pass
    k: int
        A specified number of documents for answering the user's specific question(s).
    selected_doc_first: int

    idf_use_letter: str
        Use uppercase letters, lowercase letters, or integers to mark the documents.
    use_title: int
        Whether to use title or not.
    model_name: str
        OpenAI model name.
    stage2_select_system_prompt: str
        System prompt for instruction.
    used_doc_field_in_retrieval: str
        Which filed of document to use.
    thread: "instance"
        pass
    """
    unbrowsed_docs = []

    assert idf_use_letter in ['upper', 'lower', 'int']

    while 1:
        if selected_doc_first:
            docs_concat = tmp_selected_docs + extra_docs_to_browse
        else:
            docs_concat = extra_docs_to_browse + tmp_selected_docs
        messages = create_stage2_select_prompt(questions, docs_concat, k, idf_use_letter, use_title, stage2_select_system_prompt, used_doc_field_in_retrieval)
        prompt_token_num = len(gpt2_tokenizer.tokenize(messages[0]['content'] + messages[1]['content']))
        if prompt_token_num > 3900:
            unbrowsed_docs.insert(0, extra_docs_to_browse[-1])
            extra_docs_to_browse.pop()
        else:
            break
        if len(extra_docs_to_browse) == 0:
            break

    final_docs_in_query = [docs_concat]

    if len(unbrowsed_docs) > 0:
        logger.info('before openai query, unbrowsed_docs > 0 : {}'.format(len(unbrowsed_docs)))

    def repeat_until_success_call_openai_api(func):
        def wrapper(*args, **kw):
            while True:
                result = None
                try:
                    result = func(*args, **kw)
                except openai.error.APIConnectionError as e:
                    if thread.print_error:
                        logger.info('openai connection error, so retry after sleeping 5 seconds')
                        logger.info(e)
                    time.sleep(5)
                except openai.error.RateLimitError as e:
                    logger.info(type(e))
                    logger.info(e)
                    logger.info('e._message:{}'.format(e._message))
                    if 'quota' in e._message:
                        if thread.print_error:
                            logger.info('now openai account {} runs out. so use next.'.format(thread.account[-1]))
                            logger.info(type(e))
                            logger.info(e)
                        thread.account = thread.openai_account_manager_multi_thread.get_next_account(thread.thread_id,
                                                                                                     thread.account)
                    elif "maximum context length is" in e._message:
                        unbrowsed_docs.insert(0, extra_docs_to_browse[-1])
                        extra_docs_to_browse.pop()

                        if selected_doc_first:
                            docs_concat = tmp_selected_docs + extra_docs_to_browse
                        else:
                            docs_concat = extra_docs_to_browse + tmp_selected_docs
                        final_docs_in_query[0] = docs_concat
                        messages = create_stage2_select_prompt(questions, docs_concat, k, idf_use_letter, use_title, stage2_select_system_prompt, used_doc_field_in_retrieval)
                        print('in repeat_until_success_call_openai_api, docs < 20 : {}'.format(
                            len(docs_concat)))
                        kw['messages'] = messages
                    else:
                        if True:
                            logger.info('openai rate limit error, so retry after sleeping 45 seconds')
                        time.sleep(45)
                except openai.error.AuthenticationError as e:
                    if 'This key is associated with a deactivated account' in e._message:
                        logger.info('the account {} is deactivated. so use next'.format(thread.account[-1]))
                        if thread.print_error:
                            logger.info(e)
                        thread.account = thread.openai_account_manager_multi_thread.get_next_account(thread.thread_id,
                                                                                                     thread.account)
                    else:
                        logger.info('meet unexpected AuthenticationError, so retry after sleeping 5 seconds')
                        if thread.print_error:
                            logger.info(e)
                        thread.account = thread.openai_account_manager_multi_thread.get_next_account(thread.thread_id,
                                                                                                     thread.account)
                except openai.error.InvalidRequestError as e:
                    if "maximum context length is" in e._message:
                        unbrowsed_docs.insert(0, extra_docs_to_browse[-1])
                        extra_docs_to_browse.pop()

                        if selected_doc_first:
                            docs_concat = tmp_selected_docs + extra_docs_to_browse
                        else:
                            docs_concat = extra_docs_to_browse + tmp_selected_docs
                        final_docs_in_query[0] = docs_concat
                        messages = create_stage2_select_prompt(questions, docs_concat, k, idf_use_letter, use_title, stage2_select_system_prompt, used_doc_field_in_retrieval)
                        print('in repeat_until_success_call_openai_api, docs < 20 : {}'.format(len(docs_concat)))
                        kw['messages'] = messages

                except openai.error.OpenAIError as e:
                    logger.info('meet unexpected openai error, so retry after sleeping 5 seconds')
                    logger.info(e)
                    logger.info(type(e))
                    time.sleep(3)

                except Exception as e:
                    raise e

                if result != None:
                    return result
                else:
                    pass

        return wrapper

    @repeat_until_success_call_openai_api
    def tmp_func(messages):
        return openai.ChatCompletion.create(model=model_name, messages=messages, temperature=0, max_tokens=64, api_key=thread.account[-1])

    if "gpt-3.5-turbo" in model_name:
        response = tmp_func(messages=messages)
        response = response['choices'][0]['message']['content']
    else:
        raise NotImplementedError
    response = response.split('\n')
    if len(response) > 1:
        logger.info('response has > 1 lines, so just use its first line which has the selected documents')
        logger.warning(f"response: \n{response}")
    response = response[0]

    if len(unbrowsed_docs) > 0:
        logger.info('after openai query, unbrowsed_docs > 0 : {}'.format(len(unbrowsed_docs)))

    response_document_identifiers = response.replace(',', ' ').replace('[', ' ').replace(']', ' ').strip().split()
    selected_doc_idfs = []

    docs_concat_in_openai_query = final_docs_in_query[0]
    for idf in response_document_identifiers:
        try:
            if idf_use_letter == 'upper':
                idf = letter_to_int_upper(idf)
            elif idf_use_letter == 'lower':
                idf = letter_to_int_lower(idf)
            else:
                idf = int(idf) - 1

            if idf >= len(docs_concat_in_openai_query):
                print('idf={}, response={}'.format(idf, response))
            else:
                selected_doc_idfs.append(idf)
        except:
            pass

    if len(selected_doc_idfs) != k:
        print('len(retrieved_doc_idfs) != k, k:{}, len:{},\nresponse:\n{}response_document_identifiers:\n{}'.format(k,
                                                                                                                    len(selected_doc_idfs),
                                                                                                                    response,
                                                                                                                    response_document_identifiers))

    selected_doc_idfs = selected_doc_idfs[:k]

    docs_concat_in_openai_query = final_docs_in_query[0]

    result_dict = {}

    selected_docs = []
    for idf in selected_doc_idfs:
        selected_docs.append(docs_concat_in_openai_query[idf])
    result_dict['selected_docs'] = selected_docs

    original_openai_response = response
    result_dict['original_openai_response'] = original_openai_response

    parsed_doc_idfs = selected_doc_idfs
    result_dict['parsed_doc_idfs'] = parsed_doc_idfs

    result_dict['unbrowsed_docs'] = unbrowsed_docs

    return result_dict


def iterative_select_supporting_documents_single(alce_item: Dict, 
                                                 k: int, 
                                                 window_size: int, 
                                                 reversed_browse_order: int, 
                                                 selected_doc_first: int,
                                                 idf_use_letter: str, 
                                                 use_title: int, 
                                                 model_name: str, 
                                                 stage2_select_system_prompt: str, 
                                                 used_doc_field_in_retrieval: str, 
                                                 thread: "instance", 
                                                 use_sub_questions: int=0, 
                                                 old_selected_docs: List[Dict]=None,
                                                 position: str=None,
                                                 doc_num: int=100) -> Dict:
    """Iteratively select supporting documents.
    Args
    ----
    alce_item: Dict
        Single data.
    k: int
        A specified number of documents for answering the user's specific question(s).
    window_size: int
        Context length.
    reversed_browse_order: int
        Whether to reverse the document order or not.
    selected_doc_first: int
        Whether to use the selected documents first or not.
    idf_use_letter: str
        Use uppercase letters, lowercase letters, or integers to mark the documents.
    use_title: int
        Whether to use title or not.
    model_name: str
        Which model of OpenAI to use.
    stage2_select_system_prompt: str
        System prompt for instruction.
    used_doc_field_in_retrieval: str
        Which filed of document to use in retrieval.
    thread: "instance"
        Instance of thread.
    use_sub_questions: int=0
        Whether to use sub questions for asqa.
    old_selected_docs: List[Dict]=None
        Old selected docs. May be less than 5.
    position: str=None
        Put the top-5 docs from old selected docs into the head or tail.
    doc_num: int=100
        Use top-k docs for reranking.

    Returns
    -------
    output_alce_item: Dict
        Selected docs.
    """
    output_alce_item = copy.deepcopy(alce_item)
    question = alce_item['question']
    asqa_questions = None
    if use_sub_questions and 'qa_pairs' in alce_item:
        logger.warning("Use sub questions for asqa.")
        asqa_questions = list(map(lambda x: x['question'], list(alce_item['qa_pairs'])))

    if asqa_questions != None:
        questions = asqa_questions
    else:
        questions = [question]

    docs_to_browse = copy.deepcopy(alce_item['docs'][:doc_num])
    logger.warning(f"The number of documents used for reranking is {len(docs_to_browse)}.")

    if old_selected_docs is not None and position == "head":
        logger.info("Add old selected docs into head.")
        old_selected_docs_copy = copy.deepcopy(old_selected_docs)
        docs_to_browse = old_selected_docs_copy + docs_to_browse
    elif old_selected_docs is not None and position == "tail":
        logger.info("Add old selected docs into tail.")
        old_selected_docs_copy = copy.deepcopy(old_selected_docs)
        docs_to_browse = docs_to_browse + old_selected_docs_copy

    if reversed_browse_order:
        docs_to_browse = list(reversed(docs_to_browse))

    tmp_selected_docs = []

    while len(docs_to_browse) > 0:
        # iteratively update tmp_selected_docs
        tmp_extra_docs_to_browse = docs_to_browse[:window_size - len(tmp_selected_docs)]
        docs_to_browse = docs_to_browse[window_size - len(tmp_selected_docs):]
        select_result_dict = select_k_supporting_documents(questions, tmp_selected_docs, tmp_extra_docs_to_browse, k,
                                                           selected_doc_first, idf_use_letter, use_title,
                                                           model_name, stage2_select_system_prompt, used_doc_field_in_retrieval, thread)

        tmp_selected_docs = select_result_dict['selected_docs']
        original_openai_response = select_result_dict['original_openai_response']
        parsed_doc_idfs = select_result_dict['parsed_doc_idfs']
        unbrowsed_docs = select_result_dict['unbrowsed_docs']

        docs_to_browse = unbrowsed_docs + docs_to_browse

    output_alce_item['docs'] = tmp_selected_docs

    return output_alce_item


class OpenAI_API_inp_Manager_MultiThread_Generalized:
    def __init__(self, idx_non_general_inp: List[Tuple], general_inp: Dict) -> None:
        """Class init
        Args
        ----
        idx_non_general_inp: List[Tuple]
            Data with index.
        general_inp: Dict
            Hyperparameter.
        """
        self.idx_non_general_inp = idx_non_general_inp
        assert idx_non_general_inp[0][0] == 0, 'the 1st idx_non_general_inp"s idx is not 0, maybe something error'
        self.general_inp = general_inp
        self.inp_lock = threading.Lock()
        self.progress_index = 0

        assert type(general_inp) == type({})


    def get_next_idx_inp(self) -> Union[List, None]:
        """
        Get next new data.
        """
        with self.inp_lock:
            if self.progress_index < len(self.idx_non_general_inp):
                tmp_idx = self.idx_non_general_inp[self.progress_index][0]
                tmp_non_general_inp = self.idx_non_general_inp[self.progress_index][1]
                tmp_general_inp = self.general_inp
                assert len(set(tmp_general_inp.keys()) & set(tmp_non_general_inp)) == 0, 'tmp_non_general_inp and tmp_general_inp has key overlap, must have problem'
                self.progress_index += 1
                return [tmp_idx, {**tmp_non_general_inp, **tmp_general_inp}]
            else:
                return None


class MyThread(threading.Thread):
    # todo: Adjust MyThread from calling_sliding_window to two_stage_retrieve
    def __init__(self, thread_id: int, account_manager: "instance", inp_manager: "instance", print_error: bool, pbar: tqdm.tqdm, print_finish: bool=True) -> None:
        """Class init.
        Args
        ----
        thread_id: int
            Thread id.
        account_manager: "instance"
            A manager for accounts of OpenAI.
        inp_manager: "instance"
            A manager for data.
        print_error: bool
            Whether to output error info or not.
        pbar: tqdm.tqdm
            Object of tqdm.
        print_finish: bool=True
            Whether to output ending info or not.
        """
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.openai_account_manager_multi_thread = account_manager
        self.openai_inp_manager = inp_manager
        self.account = self.openai_account_manager_multi_thread.get_next_account(self.thread_id)
        self.print_error = print_error
        self.pbar = pbar
        self.print_finish = print_finish


    def run(self):
        self.results_with_idx = []
        while True:
            tmp = self.openai_inp_manager.get_next_idx_inp()
            if tmp == None:
                if self.print_finish:
                    logger.info('thread {} finish'.format(self.thread_id))
                return
            else:
                tmp_idx = tmp[0]
                select_doc_input = tmp[1]
                result = iterative_select_supporting_documents_single(**select_doc_input, thread=self)
                if self.pbar is not None:
                    self.pbar.update(1)
                self.results_with_idx.append([tmp_idx, result])


from openai_account_manager import get_account_manager

def iterative_select_supporting_documents_multi_thread(items_to_select: List[Dict], 
                                                       general_input: Dict, 
                                                       num_threads: int, 
                                                       use_tqdm: bool=True, 
                                                       old_data: List[Dict]=None) -> List:
    """Iteratively select supporting documents in a multi-threaded manner.
    Args
    ----
    items_to_select: List[Dict]
        Candidate documents for selection.
    general_input: Dict
        Hyperparameter.
    num_threads: int
        Number of Thread.
    use_tqdm: bool
        Whether to use tqdm or not.
    old_data: List[Dict]=None
        Old data before updating query.

    Returns
    -------
    results: List
        Selected supporting documents.
    """
    new_items_to_select = []
    if old_data is None:
        logger.info("Old data is None...")
        for item in items_to_select:
            new_items_to_select.append({'alce_item': item})
    else:
        logger.info("Use old data...")
        question_to_docs = {item["question"]: item["docs"] for item in old_data}
        for item in items_to_select:
            new_items_to_select.append({'alce_item': item, "old_selected_docs": question_to_docs[item["question"]]})
    idx_items_to_select = list(enumerate(new_items_to_select))  # List[Tuple(index, item)]
    account_manager = get_account_manager(multi_thread=True)
    inp_manager = OpenAI_API_inp_Manager_MultiThread_Generalized(idx_items_to_select, general_input)

    if use_tqdm:
        pbar = tqdm.tqdm(total=len(idx_items_to_select))
    else:
        pbar = None

    thread_list = []
    for i in range(num_threads):
        thread_list.append(MyThread(i, account_manager, inp_manager, True, pbar))

    for t in thread_list:
        t.start()
    for i, t in enumerate(thread_list):
        t.join()

    results_with_idx = []

    for t in thread_list:
        results_with_idx.extend(t.results_with_idx)

    results_with_idx.sort(key=lambda x: x[0])
    results = list(map(lambda x: x[1], results_with_idx))
    return results
