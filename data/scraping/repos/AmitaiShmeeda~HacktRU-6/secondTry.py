import requests
import json
import openai

# Set your Slack API token and channel ID
slack_token = 'xapp-1-A057A9SCDA6-5261125133201-4b654cc3ae31672bad951bfa31cbed8a02634bc5e14ee6b30ca0071c4cb433e2'
channel_id = 'C0577A24AF7'

# Set your OpenAI API key
openai.api_key = "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"

# Set the URL for the Slack API
url = 'https://slack.com/api/conversations.history'

# Set the headers for the request
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {slack_token}'
}

# Set the parameters for the request
params = {
    'channel': channel_id,
    'limit': 1000
}

# Send the request to the Slack API
response = requests.get(url=url, headers=headers, params=params)

# Convert the response to JSON format
data = json.loads(response.text)

# Write the JSON data to a file
with open('conversation.json', 'w') as f:
    json.dump(data, f)

# Convert the JSON data to text format
with open('conversation.txt', 'w') as f:
    for message in data['messages']:
        if message['type'] == 'message':
            f.write(message['text'] + '\n')

# Read the conversation from the text file
with open('conversation.txt', 'r') as f:
    conversation = f.read()

# Ask chatGPT what to do with the conversation
prompt = f"What should I do with this conversation?\n\n{conversation}\n\n"

response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Print the response from chatGPT
print(response.choices[0].text)

