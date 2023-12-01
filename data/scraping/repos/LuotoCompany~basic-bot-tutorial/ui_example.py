import chainlit as cl
import os
import tempfile

from utils import build_index, build_messages

import openai

# https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "false"


async def send_msg(message: str):
    await cl.Message(
        content=message,
    ).send()


@cl.on_chat_start
async def start():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to start the discussion",
            accept=['application/pdf'],
            max_size_mb=50
        ).send()

    new_file, tmp_filename = tempfile.mkstemp()
    os.write(new_file, files[0].content)
    os.close(new_file)

    await send_msg('Indexing your file, please wait...')
    collection = build_index(tmp_filename, files[0].name)

    # Store the collection so it can be retrieved in the message callback
    cl.user_session.set("collection", collection)

    await send_msg('Ready! What do you want to know?')


@cl.on_message
async def on_message(message: cl.Message):
    user_question = message.content
    collection = cl.user_session.get("collection")

    # Seach entries from the vector database
    search_result = collection.query(
        query_texts=[user_question],
        n_results=10
    )
    context = [''.join(doclist) for doclist in search_result['documents']]

    stream = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=build_messages(user_question, context),
        temperature=0.25,
        top_p=0.2,
        max_tokens=512,
        stream=True
    )

    response = cl.Message(content="")

    # Stream the LLM's output to screen, token by token
    for txt in stream:
        content = txt.choices[0].delta.content or ""

        if (content):
            await response.stream_token(content)

    await response.send()

