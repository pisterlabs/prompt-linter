import openai
from openai import OpenAI
from dotenv.main import load_dotenv
import os
import json

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

client = OpenAI()

text_id = 2
file_id = "id_%s.txt"%text_id

assistant = client.beta.assistants.create(
    name="Market Intelligence Researcher",
    instructions="You are an experienced market intelligence researcher. Your job is to analyse, summarise and provide insights from market news",
    tools=[{"type": "retrieval"}],
    model="gpt-4-1106-preview"
)

thread = client.beta.threads.create()

#Pass the retrieval in the tools parameter of the Assistant to enable Retrieval
#Accessing the uploaded
file = client.files.create(
  file=open(file_id, "rb"),
  purpose='assistants'
)

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content = "Summarise the text in the file %s wihin 2 short paragraph",
    file_ids=[file.id]
)

run = openai.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)
print(run)


while run.status !="completed":
  run = openai.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
  )
  print(run.status)

messages = openai.beta.threads.messages.list(
  thread_id=thread.id
)

print('final result:')

result = messages.data[0].content[0].text.value

print(result)

summarised_file = open("summarised_%s"%file_id, "w")
summarised_file.write(result)
summarised_file.close()
