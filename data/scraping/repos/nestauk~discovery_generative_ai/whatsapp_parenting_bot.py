import json
import os

from threading import Thread

import openai
import s3fs

from dotenv import load_dotenv
from flask import Flask
from flask import request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from genai.eyfs import TextGenerator
from genai.eyfs import get_embedding
from genai.message_history import InMemoryMessageHistory
from genai.prompt_template import FunctionTemplate
from genai.prompt_template import MessageTemplate
from genai.streamlit_pages.utils import get_index
from genai.streamlit_pages.utils import query_pinecone


load_dotenv()
# Twilio settings
client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
# OpenAI model
LLM = "gpt-3.5-turbo"
TEMPERATURE = 0.6
openai.api_key = os.environ["OPENAI_API_KEY"]

AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
S3_PATH = os.environ["S3_BUCKET"] + "/prototypes/whatsapp-bot/logs"

pinecone_index = get_index(index_name="eyfs-index")
system_message = MessageTemplate.load("src/genai/parenting_chatbot/prompts/system.json")
filter_refs_function = FunctionTemplate.load("src/genai/parenting_chatbot/prompts/filter_refs_function.json")
filter_refs_user_message = MessageTemplate.load("src/genai/parenting_chatbot/prompts/filter_refs_user.json")
filter_refs_system_message = MessageTemplate.load("src/genai/parenting_chatbot/prompts/filter_refs_system.json")

# Initiate the Flask app
app = Flask(__name__)


def write_to_s3(key: str, secret: str, s3_path: str, filename: str, data: dict, how: str = "a") -> None:
    """
    Write data to a jsonl file in S3.

    Args:
        key (str): AWS access key ID.
        secret (str): AWS secret access key.
        s3_path (str): S3 bucket path.
        filename (str): Name of the file to write to.
        data (dict): Data to write to the file.
        how (str, optional): How to write to the file. Default is "a" for append. Use "w" to overwrite.

    """
    fs = s3fs.S3FileSystem(key=key, secret=secret)
    with fs.open(f"{s3_path}/{filename}.jsonl", how) as f:
        f.write(f"{json.dumps(data)}\n")


def read_from_s3(key: str, secret: str, s3_path: str, filename: str) -> list:
    """
    Read data from a jsonl file in S3 and return as a list of dictionaries.

    Args:
        key (str): AWS access key ID.
        secret (str): AWS secret access key.
        s3_path (str): S3 bucket path.
        filename (str): Name of the file to read from.

    Returns:
        list: A list of dictionaries, each representing a line in the jsonl file.
    """
    fs = s3fs.S3FileSystem(key=key, secret=secret)
    data_list = []

    with fs.open(f"{s3_path}/{filename}.jsonl", "r") as file:
        for line in file:
            data_list.append(json.loads(line))

    return data_list


def fetch_message_history(sender_contact: str, create_new: bool = True) -> InMemoryMessageHistory:
    """
    Find a folder on s3 corresponding to the sender; if doesn't exist, create a new folder

    Args:
        sender_contact (str): Sender's contact, follows a format 'whatsapp:+<phone number>'
        create_new (bool, optional): Whether to create a new folder if one doesn't exist. Defaults to True.

    Returns:
        InMemoryMessageHistory: Message history
    """
    message_history = InMemoryMessageHistory()
    try:
        messages = read_from_s3(
            AWS_KEY,
            AWS_SECRET,
            f"{S3_PATH}/{sender_contact}",
            "messages",
        )
        for message in messages:
            message_history.add_message(message)
        return message_history
    except FileNotFoundError as e:
        if create_new:
            write_to_s3(
                AWS_KEY,
                AWS_SECRET,
                f"{S3_PATH}/{sender_contact}",
                "messages",
                {"role": "system", "content": "Welcome to the Parenting Chatbot!"},
                how="w",
            )
            return message_history
        else:
            raise e


def send_links(link: str, my_contact: str, receiver_contact: str) -> None:
    """Generate text messages and send them to a given contact

    Args:
        links:
            Url to share
        my_contact:
            Sender's contact, follows a format 'whatsapp:+<phone number>'
        receiver_contact:
            Receiver's contact (ie, my contact), follows a format 'whatsapp:+<phone number>'
    """
    text = f"Read more: {link}"
    client.messages.create(body=text, from_=my_contact, to=receiver_contact)
    return


@app.route("/text", methods=["POST"])
def text_reply() -> str:
    """Respond to incoming messages"""
    receiver_contact = request.form.get("To")

    # Fetch message history for this sender
    sender_contact = request.form.get("From")
    message_history = fetch_message_history(sender_contact)

    # Save the incoming message to the message history
    prompt = request.form.get("Body")

    # Generate response to the message

    # Search the vector index
    search_results = query_pinecone(
        index=pinecone_index,
        encoded_query=get_embedding(prompt),
        top_n=3,
        filters={
            "source": {"$eq": "nhs_full_page"},
        },
    )

    nhs_texts = []
    nhs_urls = []
    for result in search_results:
        pred = TextGenerator.generate(
            temperature=0.0,
            messages=[filter_refs_system_message, filter_refs_user_message],
            message_kwargs={"text": result["metadata"]["text"], "question": prompt},
            functions=[filter_refs_function.to_prompt()],
            function_call={"name": filter_refs_function.name},
        )

        pred = json.loads(pred["choices"][0]["message"]["function_call"]["arguments"])["prediction"]

        if pred:
            nhs_texts.append(result["metadata"]["text"])
            nhs_urls.append(result["metadata"]["url"])

    if nhs_texts:
        nhs_texts = "\n===\n".join(nhs_texts)

    # Add references to the prompt
    prompt = f"""###NHS Start for Life references###\n{nhs_texts}\n\n###User message###\n{prompt} \n\n###Additional instructions###\nAnswer in one or two sentences, not more."""  # noqa: B950

    message_history.add_message({"role": "user", "content": prompt})
    write_to_s3(
        AWS_KEY,
        AWS_SECRET,
        f"{S3_PATH}/{sender_contact}",
        "messages",
        message_history.messages[-1],
        how="a",
    )

    response = TextGenerator.generate(
        model=LLM,
        temperature=TEMPERATURE,
        messages=message_history.get_messages(),
        message_kwargs=None,
    )
    response = response["choices"][0]["message"]["content"]
    message_history.add_message({"role": "assistant", "content": response})

    # Save message history
    write_to_s3(
        AWS_KEY,
        AWS_SECRET,
        f"{S3_PATH}/{sender_contact}",
        "messages",
        message_history.messages[-1],
        how="a",
    )
    resp = MessagingResponse()
    resp.message(response)

    # Only when incoming whatsapp message
    if (len(nhs_urls) > 0) and ("whatsapp" in receiver_contact):
        thread = Thread(target=send_links, args=[nhs_urls[0], receiver_contact, sender_contact])
        thread.start()

    return str(resp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
