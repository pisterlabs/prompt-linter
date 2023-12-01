import openai
from openai import OpenAI
from dotenv.main import load_dotenv
import os

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

client = OpenAI()

key_word = 'aluminum'

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
  file=open("news.json", "rb"),
  purpose='assistants'
)

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Using the news.json file, return the list of id of the most relevant headlines, with relevance score from 1 to 10 , somewhat related to the topic %s, even if the headline does not include that keyword. Also evaluate the sentiment score of the news, whether it is positive or negative, give the score from -1 to 1 with -1 to 0 being negative and 0 to 1 being positive. Do not explain anything. Return the result in json format" %key_word,
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

json_file = open("headlines_%s.json"%key_word, "w")
json_file.write(result)
json_file.close()

with open("headlines_%s.json"%key_word, "r") as f:
    lines = f.readlines()
with open("headlines_%s.json"%key_word, "w") as f:
    for line in lines:
        if (line.strip("\n") != "```json") and (line.strip("\n") != "```") :
            f.write(line)
