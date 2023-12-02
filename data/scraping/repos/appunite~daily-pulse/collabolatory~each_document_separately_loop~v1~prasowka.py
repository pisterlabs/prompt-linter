# This solution separately sends each document with prompt using for loop. It must be optimized in terms of prompts used and message format before using.
# Call in terminal "pip install langchain openai tiktoken requests" before running the script
# Additionally, put value for variables "folder_name" (local directory where are documents to summarize) and

import os
import requests
import json
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter

template = "Write two sentences funny summary of this document: {document}."
beginning = "Write a catchy one sentence beginning of a company-wide message which will include a summary of the previous day's decision made."
ending = "Write a catchy one sentence ending of a company-wide message which will include a summary of the previous day's decision made."

prompt = PromptTemplate(
    input_variables=["document"],
    template=template,
)

llm = OpenAI(temperature=1)
text_splitter = CharacterTextSplitter()
chain = LLMChain(llm=llm, prompt=prompt)
folder_name = ""


def updates(summaries):
    return "\n".join(str(summary) for summary in summaries)


summaries = []
with os.scandir(folder_name) as entries:
    for entry in entries:
        with open(f"{folder_name}/{entry.name}") as f:
            adrs = f.read()
        texts = text_splitter.split_text(adrs)
        summary = chain.run(texts)
        summaries.append(summary)

slack_update = f"""
  {llm(beginning)}
  {updates(summaries)}
  {llm(ending)}
"""

print(slack_update)


# send summary to Slack (using HTTP Client https://requests.readthedocs.io/en/latest/)

api_token = ""
url = "https://slack.com/api/chat.postMessage"
headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
channel = "C057MGHDX7B"  # id of prasowka-bau-adr-summaries-test

data = json.dumps({"channel": channel, "text": slack_update, "type": "mrkdwn"})

response = requests.post(url, headers=headers, data=data)
print(response.json())


# send prompts used in summary for debugging purposes as a comment in the thread

details = f"""
**beginning prompt**: {beginning}
**summary prompt**: {prompt}
**ending prompt**: {ending}
"""

data = json.dumps(
    {
        "channel": channel,
        "text": details,
        "type": "mrkdwn",
        "thread_ts": response.json()["ts"],
    }
)

response = requests.post(url, headers=headers, data=data)
print(response.json())
