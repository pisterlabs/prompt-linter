from datetime import datetime
import os
from fastapi import HTTPException, status
from sources.db_utils import connect_db
from dotenv import load_dotenv
from slack_sdk import WebClient
from prisma import Prisma
from slack_sdk.errors import SlackApiError
import uuid

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from utils.embeddings import get_embeddings


load_dotenv()


def summarize_conversation(raw_conv):
    """
    Summarize a conversation based on its raw messages
    """
    # return ""
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    message_texts = [str(msg['userId']) + ": " + msg["text"]
                     for msg in raw_conv]
    docs = [Document(page_content=text) for text in message_texts]

    prompt = """
    Write a concise summary of the following. Highlight clearly what was said, and by whom.
    "{text}"
    CONCISE SUMMARY:
    """
    prompt_template = PromptTemplate(template=prompt, input_variables=["text"])

    summarize_chain = load_summarize_chain(
        llm, chain_type="stuff", prompt=prompt_template)
    summary = summarize_chain.run(docs)
    return summary


async def get_slack_profile(user_id, client):
    """
    Fetch data from the linear API based on the query string queryStr
    """
    try:
        slack_token = os.getenv("SLACK_API_TOKEN")
        if not slack_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No slack API token found")

        try:
            # Fetch conversations from the specified channel
            result = client.users_info(user=user_id)
        except SlackApiError as e:
            print(f"Error: {e.response['error']}")
            exit(1)

        return result['user']['profile']

    except Exception as ex:
        print(ex)
        raise ex


async def find_or_create_user(slack_profile, db: Prisma, slack_user_id):
  # check if user with slackId or email exists in database
    user = await db.user.find_first(where={"email": slack_profile["email"]})

    if not user:
        # create user
        user = await db.user.create({"slackId": slack_user_id, "email": slack_profile["email"], "name": slack_profile["real_name"]})
    elif not user.slackId:
        # update the record with slackId
        user = await db.user.update(where={"email": slack_profile["email"]}, data={"slackId": slack_user_id})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error: User not found",
        )

    return user


async def get_slack_data():
    """
    Fetch data from the slack API based on the query string queryStr
    """
    try:
        db = await connect_db()
        if not db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Database connection failed")

        slack_token = os.getenv("SLACK_API_TOKEN")
        if not slack_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No slack API token found",
            )

        # Create a client instance
        client = WebClient(token=slack_token)

        response = client.conversations_list(
            types='public_channel,private_channel')
        channels = list(
            map(lambda channel: channel["id"], response["channels"]))

        # Create a list to hold the processed conversations
        processed_conversations = []
        all_raw_messages = []

        for channel in channels:
            # Specify the channel to fetch conversations from
            try:
                # Fetch conversations from the specified channel
                result = client.conversations_history(channel=channel)
            except SlackApiError as e:
                print("Error fetching result for channel: ", e)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error: {e.response['error']}",
                )

            try:
                # Fetch conversations from the specified channel
                result = client.conversations_history(channel=channel)
            except SlackApiError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error: {e.response['error']}",
                )

            # Iterate over each message in the channel's history
            for message in result["messages"]:
                print("entered messages")
                # Create a list to hold the raw messages of this conversation
                slackUrl = "https://app.slack.com/client/T05AS05S5FV/" + \
                    channel + "/" + "p" + message["ts"].replace(".", "")
                raw_messages = []
                users_in_conversation = set()

                slack_user_id = str(message["user"])
                message_id = str(message["ts"].replace(
                    ".", "")) + slack_user_id
                slack_profile = await get_slack_profile(user_id=slack_user_id, client=client)
                print("retrieved slack profile")

                # check if user with slackId or email exists in database
                user = await find_or_create_user(slack_profile, db, slack_user_id)
                users_in_conversation.add(user.id)

                # Transform the message into a raw message dictionary and add it to the list
                raw_message = {
                    # Use the timestamp as a unique ID
                    "id": message_id,
                    "text": message["text"],
                    "time": datetime.fromtimestamp(float(message["ts"])).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "userId": user.id
                }
                # if the message id has been seen before, skip it
                if await db.rawmessage.find_first(where={"id": raw_message["id"]}):
                    continue
                await db.rawmessage.create(raw_message)

                raw_messages.append(raw_message)

                # Initialize the conversation's end time with the message's timestamp
                end_time = datetime.fromtimestamp(float(message["ts"]))

                # If the message has replies, fetch them and update the conversation's end time
                if "thread_ts" in message:
                    thread_result = client.conversations_replies(
                        channel=channel, ts=message["thread_ts"])
                    if len(thread_result["messages"]) > 0:
                        slackUrl = "https://app.slack.com/client/T05AS05S5FV/" + \
                            channel + "/thread/" + \
                            channel + "-" + message["ts"]
                    for reply in thread_result["messages"]:
                        if reply["ts"] == message["ts"]:
                            continue  # Skip the message itself
                        slack_reply_user_id = str(reply["user"])
                        reply_message_id = str(reply["ts"].replace(
                            ".", "")) + slack_reply_user_id
                        slack_reply_profile = await get_slack_profile(user_id=slack_reply_user_id, client=client)
                        inner_user = await find_or_create_user(slack_reply_profile, db, slack_reply_user_id)
                        users_in_conversation.add(inner_user.id)

                        # Transform each reply into a raw message dictionary and add it to the list
                        reply_raw_message = {
                            # Use the timestamp as a unique ID
                            "id": reply_message_id,
                            "text": reply["text"],
                            "time": datetime.fromtimestamp(float(reply["ts"])).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            # Assume user ID is like 'U12345'
                            "userId": inner_user.id
                        }

                        if await db.rawmessage.find_first(where={"id": reply_raw_message["id"]}):
                            continue
                        await db.rawmessage.create(reply_raw_message)

                        raw_messages.append(reply_raw_message)

                        # Update the conversation's end time with the reply's timestamp
                        end_time = datetime.fromtimestamp(
                            float(reply["ts"])).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

                # Summarize the conversation
                print("raw messages: ", raw_messages)
                summary = summarize_conversation(raw_messages)
                print(summary)
                embed = str(get_embeddings(summary))

                # Transform the thread into a processed conversation dictionary
                processed_conversation = {
                    # Use the timestamp as a unique ID
                    "summary": summary,  # You need to implement how to generate a summary
                    "embedding": embed,
                    "slackUrl": slackUrl,
                    "startTime": datetime.fromtimestamp(float(message["ts"])).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "endTime": end_time,
                    "rawMsgs": {"connect": list(map(lambda msg: {"id": msg["id"]}, raw_messages))},
                    # Assume user ID is like 'U12345'
                    "users": {"connect": list(map(lambda user: {"id": user}, users_in_conversation))},
                }
                await db.processedconversation.create(processed_conversation)
                print("Created processed conversation")
                processed_conversations.append(processed_conversation)
                # append all raw messages to all_row_messages
                all_raw_messages.extend(raw_messages)

        data = {
            "processed_conversations": processed_conversations,
            "raw_messages": all_raw_messages
        }

        # Write the processed conversation to the database
        return data

    except Exception as ex:
        # print line number
        print("Exception getting data from slack: ", ex)
        raise ex
