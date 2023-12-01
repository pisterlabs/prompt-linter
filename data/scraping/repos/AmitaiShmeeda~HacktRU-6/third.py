import openai
import slack_sdk
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Authenticate with Slack API
client = WebClient(token="xoxb-5245319467909-5245510607157-FmgnMSyxya2AeAP1tMcPisa8")

# Listen for incoming messages from a specific channel
channel_id = "C0577A24AF7"
while True:
    try:
        response = client.conversations_history(channel=channel_id)
        messages = response["messages"]
        for message in messages:
            # Wait until you tell it to take all the conversation and send it to a file
            if "stop" in message["text"]:
                break

            # Write incoming messages to a text file
            with open("conversation.txt", "a") as f:
                f.write(message["text"] + "\n")

    except SlackApiError as e:
        print("Error: {}".format(e))

# Send conversation to OpenAI's GPT language model for processing
openai.api_key = "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"
model_engine = "davinci"

with open("conversation.txt", "r") as f:
    conversation = f.read()

response = openai.Completion.create(
    engine=model_engine,
    prompt=conversation,
    max_tokens=1024,
)

# Ask user what they want to do with the conversation
user_input = input("What do you want to do with the conversation? (summarize/opinion/background information)")

# Write output based on user input
if user_input == "summarize":
    summary = response.choices[0].text
    with open("summary.txt", "w") as f:
        f.write(summary)

elif user_input == "opinion":
    opinion = response.choices[0].text
    with open("opinion.txt", "w") as f:
        f.write(opinion)

elif user_input == "background information":
    background_info = response.choices[0].text
    with open("background_info.txt", "w") as f:
        f.write(background_info)

else:
    print("Invalid input.")
