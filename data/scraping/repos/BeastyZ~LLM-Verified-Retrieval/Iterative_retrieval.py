import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import json
import os
import concurrent.futures
import threading
from typing import List, Dict
from tqdm import tqdm
import openai
import time
import csv
import numpy as np
import torch

from openai_account_manager import get_account_manager
from utils import (
    load_embeddings,
    save_embeddings,
    get_demonstration,
    get_messages as get_messages_field
)
from llm_retrieval_related.iterative_select_supporting_documents import (
    create_stage2_select_prompt as get_messages,
    iterative_select_supporting_documents_multi_thread
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def repeat_until_success_call_openai_api(func):
    def wrapper(*args, **kwargs):
        global account_manager
        while True:
            result = None
            # account_manager = kwargs.get('account_manager', None)
            thread_id = kwargs.get('thread_id', None)
            account = account_manager.thread_to_account.get('thread_id', None)  # triad
            if account is None:
                account = account_manager.get_next_account(thread_id)
            openai.api_key = account[-1]
            try:
                result = func(*args, **kwargs)

            except openai.error.APIConnectionError as e:
                logger.info('openai connection error, so retry after sleeping 5 seconds')
                logger.info(e)
                time.sleep(5)

            except openai.error.RateLimitError as e:
                logger.info(type(e))
                logger.info(e)
                logger.info('e._message:{}'.format(e._message))
                if 'quota' in e._message:
                    logger.info('now openai account {} runs out. so use next.'.format(account[-1]))
                    logger.info(type(e))
                    logger.info(e)
                    
                    account = account_manager.get_next_account(thread_id, account)
                    account_manager.thread_to_account[thread_id] = account
                else:
                    logger.info('openai rate limit error, so retry after sleeping 60 seconds')
                    logger.info(e)
                    time.sleep(60)
                    
            except openai.error.AuthenticationError as e:
                if 'This key is associated with a deactivated account' in e._message:
                    logger.info('the account {} is deactivated. so use next'.format(account[-1]))
                    logger.info(e)
                    account = account_manager.get_next_account(thread_id, account)
                    account_manager.thread_to_account[thread_id] = account
                else:
                    logger.info('meet unexpected AuthenticationError, so retry after sleeping 5 seconds')
                    logger.info(e)
                    account = account_manager.get_next_account(thread_id, account)
                    account_manager.thread_to_account[thread_id] = account

            except openai.error.OpenAIError as e:
                logger.info('meet unexpected openai error, so retry after sleeping 5 seconds')
                logger.info(e)
                time.sleep(5)

            except Exception as e:
                raise e

            if result != None:
                return result
            else:
                pass

    return wrapper


@repeat_until_success_call_openai_api
def call_chatgpt(model: str, messages: List[Dict], thread_id: int, max_tokens: int=4096) -> str:
    """Call ChatGPT.
    Args
    ----
    model: str
        ID of the model to use.
    messages: List[Dict]
        A list of messages comprising the conversation so far.
    thread_id: int
        Thread ID
    max_tokens: int=4096
        Max tokens to generate.

    Returns
    -------
    ret: str
        Answer to prompt from ChatGPT.
    """
    if max_tokens == 4096:
        completion = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    else:
        completion = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, max_tokens=max_tokens)
    ret = completion['choices'][0]['message']['content']
    return ret


def gen_used_field_multi_thread(questions: str, 
                                prompt_style: str, 
                                doc: Dict, 
                                model_name: str = "gpt-3.5-turbo-0301", 
                                target_used_field: str = None,
                                # target_used_field: Literal["summary_use_sub", "summary_no_sub", "summary", "answer"]=None,
                                max_tokens: int = 150) -> None:
    """Generate used field by multithreading.
    Args
    ----
    questions: str
        Given questions.
    prompt_style: str
        Set the type of user's content.
    doc: Dict
        Single document.
    model_name: str="gpt-3.5-turbo-0301"
        OpenAI model.
    target_used_field: Literal["summary_use_sub", "summary_no_sub", "summary", "answer"]=None
        Which used field to generate
    max_tokens int=150.
        Max tokens to generate.
    """
    thread_id = threading.current_thread().ident
    messages = get_messages_field(questions, prompt_style, doc)
    ret = call_chatgpt(model=model_name, messages=messages, thread_id=thread_id, max_tokens=max_tokens)
    doc[target_used_field] = ' '.join(ret.strip().split())


def filter_bad_question(item: Dict, 
                        openai_model_name: str, 
                        filtration_system_prompt: str, 
                        k: int, 
                        idf_use_letter: str, 
                        use_title: int, 
                        used_doc_field: str,
                        filtration_result: str) -> None:
    """Drop bad data that documents cannot fully support the question.
    Args
    ----
    item: Dict
        Single data.
    openai_model_name: str
        ID of the model to use.
    filtration_system_prompt: str
        System prompt for instruction.
    k: int
        K number of documents to be used.
    idf_use_letter: str
        Use uppercase letters, lowercase letters, or integers to mark the documents.
    use_title: int
        Whether to use title or not.
    used_doc_field: str
        Which filed of document to use.
    filtration_result: str=Literal["judgment", "threshold"]
        Field used for saving.
    """
    global account_manager

    if 'qa_pairs' in item:
        questions = list(map(lambda x: x['question'], list(item['qa_pairs'])))
    else:
        question = item['question']
        questions = [question]
    # Using top-k documents.
    docs = item["docs"][:k]

    thread_id = threading.current_thread().ident
    messages = get_messages(questions, docs, k, idf_use_letter, use_title, filtration_system_prompt, used_doc_field)
    result = call_chatgpt(model=openai_model_name, messages=messages, thread_id=thread_id)
    item[filtration_result] = result


def update_query(user_query: str, system_prompt: str, model: str, d: Dict, max_tokens: int=4096) -> None:
    """Update query by ChatGPT.
    Args
    ----
    user_query: str
        Query for ChatGPT.
    system_prompt: str
        System prompt for ChatGPT.
    model: str
        ID of the model to use.
    d: Dict
        Single data.
    max_tokens: int=4096
        Max tokens to generate.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    thread_id = threading.current_thread().ident
    query = call_chatgpt(model=model, messages=messages, thread_id=thread_id, max_tokens=max_tokens)
    d["update_query_using_missing_info_from_question_and_psgs"] = query


def bm25_sphere_retrieval(data: List[Dict], system_prompt: str=None, args: object=None) -> None:
    """Use retriever of bm25 to do retrieval on the corpus of sphere.
    Args
    ----
    data: List[Dict]
        A list of data used for retrieval.
    system_prompt: str=None
        Instruction for ChatGPT when updating query.
    args: object=None
        Parameters.
    """
    if args.update_query_using_missing_info_from_question_and_psgs and system_prompt is not None:
        logger.info("update_query_using_missing_info_from_question_and_psgs")

        # Update query using multi-thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.thread_num) as executor:
            futures = []
            for d in data:
                # Get question(s)
                questions = d["question"]

                psgs = []
                for index, doc in enumerate(d["docs"]):
                    psgs.append(f"{index + 1}.{doc['text']}")
                psgs = "\n".join(psgs)

                user_query = f"Question:\n{questions}\nAnswering Passages:\n{psgs}"
                future = executor.submit(
                    update_query,
                    user_query=user_query,
                    system_prompt=system_prompt,
                    model=args.openai_model_name,
                    d=d
                )
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.thread_num) as executor:
            futures = []
            for d in data:
                questions = d["question"]
                user_query = f'Please write a passage to answer the question.\nQuestion: {questions}\nPassage: '

                future = executor.submit(
                    update_query,
                    user_query=user_query,
                    system_prompt="You are a helpful assistant.",
                    model=args.openai_model_name,
                    d=d,
                )
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()


def bge_wiki_retrieval(data: List[Dict], system_prompt: str=None, args: object=None) -> None:
    """Use retriever of bge to do retrieval on the corpus of wiki.
    Args
    ----
    data: List[Dict]
        A list of data used for retrieval.
    system_prompt: str=None
        Instruction for ChatGPT when updating query.
    args: object=None
        Parameters.
    """
    if args.update_query_using_missing_info_from_question_and_psgs and system_prompt is not None:
        logger.info("update_query_using_missing_info_from_question_and_psgs")
        
        # Update query using multi-thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.thread_num) as executor:
            futures = []
            for d in data:
                # Get question(s)
                if args.use_sub_questions:
                    asqa_questions = list(map(lambda x: x['question'], list(d['qa_pairs'])))
                    questions = "\n".join(asqa_questions)
                else:
                    questions = d["question"]

                # Get passages.
                psgs = []
                for index, doc in enumerate(d["docs"]):
                    psgs.append(f"{index + 1}.{doc['text']}")
                psgs = "\n".join(psgs)
                
                user_query = f"Question:\n{questions}\nAnswering Passages:\n{psgs}"
                future = executor.submit(
                    update_query,
                    user_query=user_query,
                    system_prompt=system_prompt,
                    model=args.openai_model_name,
                    d=d
                )
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()

    # Load questions
    questions = []
    for d in data:
        if args.update_query_using_missing_info_from_question_and_psgs and system_prompt is not None:
            if "update_query_using_missing_info_from_question_and_psgs" not in d:
                raise ValueError
            query = d["update_query_using_missing_info_from_question_and_psgs"]
            questions.append(query)
        else:
            if args.use_sub_questions:
                logger.info("Load sub questions.")
                asqa_questions = list(map(lambda x: x['question'], list(d['qa_pairs'])))
                questions.append("\n".join(asqa_questions))
            else:
                questions.append(d["question"])

    # Handle question togather and split the result of question encoding into several batches.
    from tqdm.autonotebook import trange
    q_embeddings = bge_retriever.encode_queries(questions)
    matrix_batch_size = 128
    len_p_embeddings = len(p_embeddings)
    docs_list = []
    for start_index in trange(0, len(q_embeddings), matrix_batch_size):
        embeddings_batch = q_embeddings[start_index: start_index + matrix_batch_size]
        embeddings_batch = torch.tensor(embeddings_batch).cuda()

        p_embeddings_half = torch.tensor(p_embeddings[:len_p_embeddings // 2]).cuda()
        scores_1 = torch.matmul(embeddings_batch, p_embeddings_half.t())
        del p_embeddings_half
        torch.cuda.empty_cache()

        p_embeddings_half = torch.tensor(p_embeddings[len_p_embeddings // 2:]).cuda()
        scores_2 = torch.matmul(embeddings_batch, p_embeddings_half.t())
        del p_embeddings_half
        torch.cuda.empty_cache()

        scores = torch.cat((scores_1, scores_2), dim=1)
        assert scores.shape[1] == len_p_embeddings
        # Save top-k documents.
        values, indices = torch.topk(scores, args.top_k_retrieval)
        for i, doc_idx in enumerate(indices):
            docs = []
            for j, idx in enumerate(doc_idx):
                title, text = documents[idx.item()].split("\n")
                docs.append({"id": str(idx.item() + 1), "title": title, "text": text, "score": values[i][j].item()})
            docs_list.append(docs)

    # Save retrieved top-k docs
    for d_id in range(len(data)):
        data[d_id]["docs"] = docs_list[d_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative retrieval.")
    parser.add_argument("--max_iteration", type=int, default=4)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--thread_num", type=int, default=10)
    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--use_sub_questions", type=int, required=True)
    parser.add_argument("--use_title", type=int, required=True)
    parser.add_argument("--used_doc_field", type=str, required=True)
    # Args for retrieval
    parser.add_argument("--input_file", type=str, required=True, help="Path to the original input file for the first retrieval.")
    parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--top_k_retrieval", type=int, default=100)
    parser.add_argument("--update_prompt_file", type=str, required=True)
    parser.add_argument("--update_query_using_missing_info_from_question_and_psgs", type=int, help="Whether to use this function or not.")
    parser.add_argument("--corpus_path", type=str, required=True)
    # Args for generating used field.
    parser.add_argument("--top_k_field", type=int, default=50)
    parser.add_argument("--prompt_style", type=str, choices=['summary', 'extraction', 'answer'], required=True)
    parser.add_argument('--target_used_field', type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=4096)
    # Args for reranker
    parser.add_argument("--reranker", type=str, default="llm-select")
    parser.add_argument("--position", type=str, required=True, help="Put the 5 old selected docs into the head or tail.")
    parser.add_argument("--top_k_reranking", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--reranking_prompt_file", type=str, required=True)
    parser.add_argument('--doc_num', type=int, default=100, help="Use top-k docs for reranking.")
    # Args for filtration
    parser.add_argument('--demo_file', type=str, default=None, help="Path to demonstration file.")
    parser.add_argument("--top_k_filtration", type=int, default=5)
    parser.add_argument("--idf_use_letter", type=str, default="int")
    parser.add_argument("--filtration_prompt_file", type=str, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--filtration_method", type=str, choices=["threshold", "judgment"])
    args = parser.parse_args()
    
    if args.filtration_method == "threshold":
        assert args.threshold is not None and args.filtration_prompt_file == "prompt14_score"

    # Save args
    args_dict = vars(args)
    args_dir = f'iter_retrieval_50/{args.dataset_name}_input'
    os.makedirs(args_dir, exist_ok=True)
    with open(f"{args_dir}/args_{args.retriever}_max_iteration-{args.max_iteration}_{args.update_prompt_file}_{args.position}.json", 'w') as f:
        json.dump(args_dict, f, indent=4)

    retrieval_dir = f"iter_retrieval_50/{args.dataset_name}_max-{args.max_iteration}_{args.retriever}-{args.update_prompt_file}"
    os.makedirs(retrieval_dir, exist_ok=True)
    field_dir = f"iter_retrieval_50/{args.dataset_name}_max-{args.max_iteration}_{args.retriever}-{args.update_prompt_file}_{args.target_used_field}"
    os.makedirs(field_dir, exist_ok=True)
    reranker_dir = f"iter_retrieval_50/{args.dataset_name}_max-{args.max_iteration}_{args.retriever}-{args.update_prompt_file}_{args.target_used_field}_{args.reranker}-{args.position}"
    os.makedirs(reranker_dir, exist_ok=True)
    filtration_dir = f"iter_retrieval_50/{args.dataset_name}_max-{args.max_iteration}_{args.retriever}-{args.update_prompt_file}_{args.target_used_field}_{args.reranker}-{args.position}_filtration"
    os.makedirs(filtration_dir, exist_ok=True)
    
    # Get OpenAI account manager
    account_manager = get_account_manager('openai_account_files/accounts.txt', 'openai_account_files/used.txt', multi_thread=True)

    if args.dataset_name == "eli5":
        logger.info("Load bm25 index (only sphere), this may take a while... ")
        # For single process
        # searcher = LuceneSearcher(args.corpus_path)
    elif args.dataset_name == "asqa" or args.dataset_name == "qampari":
        if args.retriever in ["bge-large-en-v1.5", "bge-base-en-v1.5"]:
            from FlagEmbedding import FlagModel

            logger.info(f"Load BAAI/{args.retriever} model.")
            bge_retriever = FlagModel(f"BAAI/{args.retriever}", 
                                query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                                use_fp16=True)
            
             # Load documents
            logger.info("Load documents.")
            documents = []
            with open(args.corpus_path) as f:
                reader = csv.reader(f, delimiter="\t")
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    documents.append(row[2] + "\n" + row[1])

            # model_name = args.retriever.split("/")[1]
            if os.path.exists(f"./embedding/psgs_w100_embedding_{args.retriever}.pkl"):
                logger.info("Load local embeddings.")
                p_embeddings = load_embeddings(f"./embedding/psgs_w100_embedding_{args.retriever}.pkl")
            else:
                logger.info("Build embeddings.")
                p_embeddings = bge_retriever.encode(documents, batch_size=256)
                save_embeddings(p_embeddings, f"./embedding/psgs_w100_embedding_{args.retriever}.pkl")

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    for iter in range(args.max_iteration):
        # 1. Retrieval stage
        logger.info("Start retrieval.")
        retrieval_path = f"{retrieval_dir}/retrieval_output_iter-{iter}.json"
        if os.path.exists(retrieval_path):
            logger.warning(f"Retrieval output in iteration {iter} already exists.")
        else:
            if iter > 0:
                logger.info(f"iter:{iter} => Update query and then start retrieval using new query.")
                update_system_prompt = ''.join(open('llm_retrieval_prompt_drafts/{}.md'.format(args.update_prompt_file)).readlines())
                with open(f"{filtration_dir}/filtration_output_iter-{iter - 1}_no.json") as f:
                    data = json.load(f)

                if args.dataset_name == "asqa":
                    bge_wiki_retrieval(data, update_system_prompt, args)
                elif args.dataset_name == "qampari":
                    bge_wiki_retrieval(data, update_system_prompt, args)
                elif args.dataset_name == "eli5":
                    # Get new prompt
                    bm25_sphere_retrieval(data, update_system_prompt, args)
                    queries = []
                    for d in data:
                        queries.append(d["update_query_using_missing_info_from_question_and_psgs"])

                    logger.info("Start bm25 retrieval using multi-process.")
                    from multi_process.bm25_multi_process import BM25MultiProcess
                    bm25_multiprocess = BM25MultiProcess(args.corpus_path)
                    pool = bm25_multiprocess.start_multi_process_pool(process_num=15)
                    docs_list = bm25_multiprocess.retrieve_multi_process(queries, pool)
                    bm25_multiprocess.stop_multi_process_pool(pool)
                    for i in range(len(data)):
                        data[i]["docs"] = docs_list[i]
                else:
                    raise NotImplementedError
                
            else:
                logger.info("First time doing a retrieval.")
                with open(args.input_file) as f:
                    data = json.load(f)

                if args.dataset_name == "asqa":
                    bge_wiki_retrieval(data, None, args)
                elif args.dataset_name == "qampari":
                    bge_wiki_retrieval(data, None, args)
                elif args.dataset_name == "eli5":
                    bm25_sphere_retrieval(data, None, args)
                else:
                    raise NotImplementedError
                
            with open(retrieval_path, "w") as f:
                json.dump(data, f, indent=4)
        logger.info("Finish retrieval.")

        # 2. Generating used field stage
        logger.info("Start generating used field.")
        field_path = f"{field_dir}/field_output_iter-{iter}.json"
        if os.path.exists(field_path):
            logger.warning(f"Field output in iteration {iter} already exists.")
        else:
            with open(retrieval_path) as f:
                data = json.load(f)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.thread_num) as executor:
                futures = []
                for d in data:
                    if args.use_sub_questions and 'qa_pairs' in d:
                        questions = list(map(lambda x: x['question'], list(d['qa_pairs'])))
                    else:
                        questions = [d["question"]]
                    questions = '\n'.join(questions)

                    for doc in d["docs"][:args.top_k_field]:
                        future = executor.submit(
                            gen_used_field_multi_thread,
                            questions,
                            args.prompt_style,
                            doc,
                            args.openai_model_name, 
                            args.target_used_field, 
                            max_tokens=args.max_tokens
                        )
                        futures.append(future)
            
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    future.result()
                
            with open(field_path, "w") as f:
                json.dump(data, f, indent=4)
        logger.info("Finish generating used field.")

        # 3. Reranking stage
        logger.info("Start reranking.")
        reranking_path = f"{reranker_dir}/reranker_output_iter-{iter}.json"
        if os.path.exists(reranking_path):
            logger.warning(f"Reranker output in iteration {iter} already exists.")
        else:
            reranking_system_prompt = ''.join(open('llm_retrieval_prompt_drafts/{}.md'.format(args.reranking_prompt_file)).readlines())
            select_gpt_hyper_parameter = {
                'k': args.top_k_reranking, 'window_size': args.window_size, 'reversed_browse_order': 0, 'selected_doc_first': 1,
                'idf_use_letter': "int", 'use_title': args.use_title, 'model_name': args.openai_model_name, 
                'used_doc_field_in_retrieval': args.used_doc_field, "use_sub_questions": args.use_sub_questions, 
                "position": args.position, "stage2_select_system_prompt": reranking_system_prompt, "doc_num": args.doc_num
            }
            if iter == 0:
                old_data = None
            else:
                with open(f"{filtration_dir}/filtration_output_iter-{iter - 1}_no.json") as f:
                    old_data = json.load(f)

            with open(field_path) as f:
                data = json.load(f)
                    
            reranked_data = iterative_select_supporting_documents_multi_thread(data, select_gpt_hyper_parameter, args.thread_num, True, old_data=old_data)
            with open(reranking_path, "w") as f:
                json.dump(reranked_data, f, indent=4)
        logger.info("Finish reranking.")
        # If it is the last iteration, skip filtration and exit loop.
        if iter == args.max_iteration - 1:
            break

        # 4. Filtration stage
        logger.info("Start filtration.")
        filtration_path = f"{filtration_dir}/filtration_output_iter-{iter}_yes.json"
        if os.path.exists(filtration_path):
            logger.warning(f"Yes/No filtration output in iteration {iter} already exists.")
        else:
            with open(reranking_path) as f:
                data = json.load(f)

            filtration_system_prompt = ''.join(open('llm_retrieval_prompt_drafts/{}.md'.format(args.filtration_prompt_file)).readlines())
            if args.demo_file is not None:
                logger.warning("Use demonstration for filtration.")
                assert "with_demo" in args.filtration_prompt_file
                
                with open(args.demo_file) as f:
                    demo_data = json.load(f)
                demos = get_demonstration(demo_data)
                filtration_system_prompt = filtration_system_prompt.replace("{Demo}", demos)

            if args.filtration_method == "judgment":
                filtration_result = "judgment"
            else:
                raise NotImplementedError

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.thread_num) as executor:
                futures = []
                for d in data:
                    future = executor.submit(
                        filter_bad_question,
                        d,
                        args.openai_model_name,
                        filtration_system_prompt,
                        args.top_k_filtration,
                        args.idf_use_letter,
                        args.use_title,
                        args.used_doc_field,
                        filtration_result
                    )
                    futures.append(future)
            
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    future.result()

            wanted_data = []
            dropped_data = []
            for d in data:
                # Use JudgmentGPT
                if args.filtration_method == "judgment":
                    if "[YES]" in d["judgment"]:
                        wanted_data.append(d)
                    else:
                        dropped_data.append(d)

                else:
                    raise NotImplementedError
            
            with open(filtration_path, "w") as f:
                json.dump(wanted_data, f, indent=4)

            # All data meets requirements
            if len(dropped_data) == 0:
                logger.info("All data meets requirements, so exit.")
                break
            
            with open(f"{filtration_dir}/filtration_output_iter-{iter}_no.json", "w") as f:
                json.dump(dropped_data, f, indent=4)
        logger.info("Finish filtration.")

    # Save final data for run & eval stage
    logger.info("Start Saving all data for run & eval")
    all_data = []
    for iter in range(args.max_iteration):
        if iter == args.max_iteration - 1:
            with open(f"{reranker_dir}/reranker_output_iter-{iter}.json") as f:
                all_data += json.load(f)
        else:
            with open(f"{filtration_dir}/filtration_output_iter-{iter}_yes.json") as f:
                all_data += json.load(f)

    save_dir = f"iter_retrieval_50/{args.dataset_name}_final_data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/final_data_{args.retriever}_max_iteration-{args.max_iteration}_{args.update_prompt_file}_{args.position}.json"
    if os.path.exists(save_path):
        logger.warning(f"{args.dataset_name} final data already exist.")
    else:
        with open(save_path, "w") as f:
            json.dump(all_data, f, indent=4)
    logger.info("Get it all done.")
