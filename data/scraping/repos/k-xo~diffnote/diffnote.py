import os
import tiktoken
from openai import OpenAI
import time


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
tokenizer = tiktoken.encoding_for_model("gpt-4-1106-preview")


def generate_diffnote(git_diff):
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=git_diff,
    )

    assistant = client.beta.assistants.retrieve("asst_NYGwmk5P299OoPsWqts5HGMA")
    run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant.id
    )

    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(0.2)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    assistant_response = ""
    for i in messages:
        if i.role == "assistant":
            assistant_response = i.content[0].text.value
            break

    return assistant_response


def diffnote():
    git_diff = os.popen("git diff").read()

    commit_msg = generate_diffnote(git_diff)
    commit_msg = commit_msg.replace("‘", "").replace("’", "").replace("`", "")

    os.system(f"git commit -am '{commit_msg}'")


if __name__ == "__main__":
    diffnote()
