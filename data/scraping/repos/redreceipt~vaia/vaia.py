import json
import os
import time
from datetime import datetime

import schedule
from dotenv import load_dotenv
from openai import OpenAI
from twilio.rest import Client

load_dotenv()

openai_client = OpenAI()

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
to_number = os.environ["TWILIO_TO_NUMBER"]
from_number = os.environ["TWILIO_FROM_NUMBER"]
twilio_client = Client(account_sid, auth_token)


def create_todo(user_input, messages=[]):
    if not len(messages):
        messages = [
            {
                "role": "system",
                "content": f"""
                        You are a helpful assistant designed to output JSON.
                        You are given a user input and asked to turn it into a reminder.
                        The users current time is {datetime.now().strftime("%H:%M")}.
                        Try to make assumptions about the reminder and reminder time
                        from the initial user input. If you don't have enough information
                        to create the reminder, keep asking questions
                        until you have enough information, and then confirm with the user.
                        Once the user confirms, output the final message in JSON format
                        with reminder, reminder_time, and confirmation keys.
                        Format the reminder_time as HH:MM in 24H format.
                        The confirmation key should be the message you are giving to the user confirming
                        the details of the reminder in a friendly, concise, conversational format,
                        well suited for an SMS response.
                        The reminder key should be reformatted to be in the form of reminding the user
                        of the event they requested.
                        For example, if the user input is "Remind me to call my mom tomorrow at 9am",
                        the reminder key should be "Call your mom" and the reminder_time key should be "09:00".
                    """,
            },
        ]

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.decoder.JSONDecodeError:
        content = response.choices[0].message.content
        messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )
        new_input = input(f"{content} \n\n> ")
        return create_todo(new_input, messages)


def add_todo(reminder, reminder_time, confirmation):
    twilio_client.messages.create(body=confirmation, from_=from_number, to=to_number)

    def job():
        twilio_client.messages.create(body=reminder, from_=from_number, to=to_number)
        return schedule.CancelJob

    schedule.every().day.at(reminder_time, "America/New_York").do(job)


if __name__ == "__main__":
    prompt = input("Prompt: ")
    todo = create_todo(prompt)
    print(todo)
    add_todo(todo["reminder"], todo["reminder_time"], todo["confirmation"])
    while True:
        schedule.run_pending()
        time.sleep(1)
