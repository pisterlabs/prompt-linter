from . import (
    OPEN_AI_MODEL_TYPE,
    DEFAULT_IMAGE_MODEL,
    agents_dir,
    files_dir,
    threads_dir,
)
from openai import OpenAI
import os
import logging
import click
import time
from typing import Optional, List
from pathlib import Path
import shutil
from datetime import datetime

JSON_RESPONSE_TYPE = {"type": "json_object"}
TEXT_RESPONSE_TYPE = {"type": "text"}
ALL_TOOLS_AVAILABLE = []
MAX_AGENT_LIST = 100
thread_separator = "---"
agent_separator = "---"
filename_separator = "---"
DEFAULT_TOOLS = [{"type": "retrieval"}]
logger = logging.getLogger("oaicli")

client = OpenAI()

MAX_RUN_TIME = 120


def _get_assistant_path(my_assistant):
    return f"{agents_dir}/{my_assistant.id}"


def get_assistants(after=None):
    assistants = client.beta.assistants.list(
        order="desc",
        limit=MAX_AGENT_LIST,
        after=after,
    )
    for assistant in assistants:
        yield assistant
    if assistants.has_more:
        click.echo(f"There are more, but we stop at {MAX_AGENT_LIST}")


def create_assistant_wrapper(
    name: str, instructions: str, tools: Optional[List[str]] = None
):
    my_assistant = client.beta.assistants.create(
        instructions=instructions,
        name=name,
        tools=DEFAULT_TOOLS,
        model=OPEN_AI_MODEL_TYPE,
    )

    os.makedirs(_get_assistant_path(my_assistant), exist_ok=True)

    return my_assistant


def save_instructions(my_assistant, content):
    os.makedirs(_get_assistant_path(my_assistant), exist_ok=True)
    filepath = f"{_get_assistant_path(my_assistant)}/instructions.txt"
    file_object = open(filepath, "w")
    file_object.write(content)


def load_instructions(my_assistant):
    os.makedirs(_get_assistant_path(my_assistant), exist_ok=True)
    filepath = f"{_get_assistant_path(my_assistant)}/instructions.txt"
    file_object = open(filepath, "r")
    return file_object.read()


def list_threads():
    # seems theres no api call to list threads? can't be named?
    for first_level in os.walk(threads_dir):
        thread_dir = first_level[1]
        break
    return [thread.split(thread_separator) for thread in thread_dir]


def _get_thread_directory(thread):
    return f"{threads_dir}/{thread.metadata['name']}{thread_separator}{thread.id}"


def create_thread(thread_name: str):
    metadata = {"name": thread_name}
    empty_thread = client.beta.threads.create(metadata=metadata)
    thread_id = empty_thread.id
    click.echo(f"created thread {thread_id}.")
    os.makedirs(_get_thread_directory(empty_thread), exist_ok=True)
    return (thread_name, thread_id)


def save_local_message(thread_message: str, role: str):
    # TODO save content locally in thread directory
    thread = client.beta.threads.retrieve(thread_message.thread_id)
    thread_directory = _get_thread_directory(thread)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H:%M:%S")
    local_path = f"{thread_directory}/{timestamp}-{role}-{thread_message.id}.txt"
    file_object = open(local_path, "w")
    all_content = ""
    for _content in thread_message.content:
        all_content += _content.text.value
    file_object.write(all_content)


def create_message(
    message_content: str, thread_name: str, thread_id: str, file_ids=None
):
    role = "user"
    if not file_ids:
        file_ids = []
    thread_message = client.beta.threads.messages.create(
        thread_id, role=role, content=message_content, file_ids=file_ids
    )
    save_local_message(thread_message, role=role)
    return thread_message


def create_run(thread_id, assistant_id):
    return client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant_id
    )


def get_messages(thread_id):
    thread_messages = client.beta.threads.messages.list(thread_id)
    return thread_messages.data


def wait_for_or_cancel_run(thread_id, run_id, max_run_time=MAX_RUN_TIME):
    timeout_is_ok = True
    total_time = 0

    CHECK_INCREMENT = 2
    click.echo(f"Running for a max of {max_run_time} seconds.")
    while timeout_is_ok:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status != "completed":
            click.echo(f"Job is {run.status}. {total_time} s passed.", nl=False)
            time.sleep(CHECK_INCREMENT)
            click.echo("\r", nl=False)
            total_time += CHECK_INCREMENT
            if run.status in ["cancelled", "failed", "expired"]:
                return
            if total_time >= max_run_time:
                run = client.beta.threads.runs.cancel(
                    thread_id="thread_abc123", run_id="run_abc123"
                )
        elif run.status == "completed":
            return True


def _get_local_filepath(file):
    new_filename = f"{file.id}{filename_separator}{file.filename}"
    return f"{files_dir}/{new_filename}"


def delete_file(file_id: str):
    client.files.delete(file_id)


def delete_assistant(assistant_id):
    client.beta.assistants.delete(assistant_id)


def upload_file(local_filepath: str):
    file = client.files.create(file=open(local_filepath, "rb"), purpose="assistants")
    file_path = Path(local_filepath)
    filename = file_path.name

    if filename != file.filename:
        click.echo(f"warning {filename} != file.filename")

    shutil.copyfile(local_filepath, _get_local_filepath(file))
    return file


def list_files():
    return client.files.list()


def download_all_files():
    # TODO -  option for file ids in use, not in use
    all_files = list_files()
    for file in all_files:
        local_path = _get_local_filepath(file)
        if not os.path.exists(local_path):
            content = client.files.retrieve_content(file.id)
            file_object = open(local_path, "w")
            file_object.write(content)
            click.echo(f"{file.filename}\t\t({file.id}) to {local_path}")
        else:
            click.echo(f"{file.filename}\t\t({file.id}) exists locally")


def list_all_files():
    # TODO -  option for file ids in use, not in use
    all_files = list_files()
    for file in all_files:
        local_path = _get_local_filepath(file)
        dt = datetime.fromtimestamp(file.created_at)
        created = dt.strftime("%Y-%m-%d-%H:%M:%S")
        file_description = f"{file.filename}\t\t({created} {file.purpose} {file.id})"
        if not os.path.exists(local_path):
            click.echo(f"{file_description}) does not exist locally")
        else:
            click.echo(f"{file_description}) exists locally")


# def vision_url(prompt: str, image_url: str):
#     response = client.chat.completions.create(
#         model=DEFAULT_IMAGE_MODEL,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": image_url,
#                             "detail": "low",
#                         },
#                     },
#                 ],
#             }
#         ],
#         max_tokens=300,
#     )

#     return response
