# Reddit: https://www.reddit.com/r/ChatGPTPro/comments/14v3y03/chatgpt_code_interpreter_for_github_repo_analysis/
# Main Tutorial: https://python.langchain.com/docs/use_cases/code/twitter-the-algorithm-analysis-deeplake
import json
import os
import datetime
from typing import List

from fastapi.exceptions import HTTPException

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
import tiktoken

llm = OpenAI(temperature=0)

from repochat.config import settings

# Define the path to the chat logs folder
chat_logs_folder = "./repochat/chat_logs"
if not os.path.exists(chat_logs_folder):
    os.makedirs(chat_logs_folder)

# Define the path to the old chat logs folder (move all but latest into old)
old_chat_logs_folder = os.path.join(chat_logs_folder, "old")
if not os.path.exists(old_chat_logs_folder):
    os.makedirs(old_chat_logs_folder)

# Get the current timestamp
# chat_log_filename = f"chat_{timestamp}.md"
timestamp = datetime.datetime.now().strftime("%m.%d.%Y_%H%M")
current_date = datetime.datetime.now().strftime("%m.%d.%Y")
chat_log_filename = f"chat_{current_date}.md"

embeddings = OpenAIEmbeddings(disallowed_special=())

db = DeepLake(
    dataset_path=f"hub://{settings.deeplake_username}/{settings.deeplake_dataset_name}",
    read_only=True,
    embedding_function=embeddings,
)


async def ask_questions(
    questions,
    gpt4_on: bool = False,
    chat_history_file: str = "./repochat/chat_logs/chat_history.json",
    test_mode: bool = False,
    pass_chat_history: int = 3,
    path_filter_layer_1: List[str] = [],
    path_filter_layer_2: List[str] = [],
):
    global db
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    # Load chat history from the JSON file (if it exists)
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r") as file:
            chat_history = json.load(file)
    else:
        chat_history = []

    # https://openai.com/pricing
    # model = ChatOpenAI(model_name="gpt-3.5-turbo")
    model_name = "gpt-3.5-turbo-16k"
    max_tokens_limit = 16000
    if gpt4_on:
        model_name = "gpt-4"
        max_tokens_limit = 8193
    model = ChatOpenAI(model_name=model_name)  # switch to 'gpt-4'

    # calculate excess tokens for document retrieval
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    # remaining tokens works per docs on max_tokens: https://github.com/hwchase17/langchain/blob/master/langchain/chains/conversational_retrieval/base.py
    tiktoken_encoding = tiktoken.encoding_for_model(model_name)
    question_tokens = 0
    chat_history_tokens = 0
    for question in questions:
        question_tokens += len(tiktoken_encoding.encode(question))
    if pass_chat_history > 0:
        for q_and_a in chat_history[-pass_chat_history:]:
            chat_history_tokens += len(tiktoken_encoding.encode(q_and_a["question"]))
            chat_history_tokens += len(tiktoken_encoding.encode(q_and_a["answer"]))

    used_tokens = question_tokens + chat_history_tokens
    remaining_tokens = max_tokens_limit - used_tokens
    print(f"pass_chat_history: {pass_chat_history}")
    print(f"max_tokens_limit: {max_tokens_limit}")
    print(f"question_tokens: {question_tokens}")
    print(f"chat_history_tokens: {chat_history_tokens}")
    print(f"used_tokens: {used_tokens}")
    print(f"remaining_tokens: {remaining_tokens}")

    if test_mode:
        for question in questions:
            answer = "This is a test answer. ```python print('hello world')```"
            chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "timestamp": timestamp,
                }
            )
    else:
        # You can also specify user defined functions using Deep Lake filters
        # https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.filter
        def combined_filters(x):
            metadata = x["metadata"].data()["value"]
            path_filter_layer_1_passes = any(
                path_filter in metadata["source"] for path_filter in path_filter_layer_1
            )
            if not path_filter_layer_1_passes:
                return False

            # If it passes filter layer 1, check if it also passes filter layer 2
            if path_filter_layer_2:
                path_filter_layer_2_passes = any(
                    metadata["source"].endswith(path_filter)
                    for path_filter in path_filter_layer_2
                )
                return path_filter_layer_2_passes

            return True  # if no path_filter_layer_2 is provided, return True if path_filter_layer_1 passes

        if path_filter_layer_1:
            retriever.search_kwargs["filter"] = combined_filters

        qa = ConversationalRetrievalChain.from_llm(
            model,
            retriever=retriever,
            max_tokens_limit=remaining_tokens,
        )
        for question in questions:
            if pass_chat_history > 0:
                chat_history_list = [
                    (q_and_a["question"], q_and_a["answer"])
                    for q_and_a in chat_history[-pass_chat_history:]
                ]
            else:
                chat_history_list = []

            try:
                result = qa({"question": question, "chat_history": chat_history_list})
                answer = result["answer"]
                chat_history.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "timestamp": timestamp,
                    }
                )
            except Exception as e:
                print(e)
                raise HTTPException(status_code=500, detail=str(e))

    # Write the chat log to the current folder
    with open(os.path.join(chat_logs_folder, chat_log_filename), "w") as file:
        for entry in reversed(chat_history):
            # Only include chats from the current day
            timestamp_date = entry["timestamp"].split("_")[0]
            if timestamp_date != current_date:
                pass
            else:
                question = entry["question"]
                answer = entry["answer"]
                file.write(f"**Question**: {question}\n\n")
                file.write(f"**Answer**: {answer}\n\n")

    # Move all other chat logs to the "old" folder
    for filename in os.listdir(chat_logs_folder):
        if filename != chat_log_filename and not os.path.isdir(
            os.path.join(chat_logs_folder, filename)
        ):
            src = os.path.join(chat_logs_folder, filename)
            dst = os.path.join(old_chat_logs_folder, filename)
            os.rename(src, dst)

    # Save the chat history to the JSON file after each question
    with open(chat_history_file, "w") as file:
        json.dump(chat_history, file)

    return chat_history
