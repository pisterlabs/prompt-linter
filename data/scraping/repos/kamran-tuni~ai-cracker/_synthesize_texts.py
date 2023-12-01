import shutil
import openai
from openai import OpenAI
from dotenv.main import load_dotenv
import os

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

client = OpenAI()

concat_file = 'concat_summarised.txt'

with open(concat_file,'wb') as wfd:
    for f in ['summarised_id_6.txt','summarised_id_9.txt','summarised_id_10.txt']:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)

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
  file=open(concat_file, "rb"),
  purpose='assistants'
)

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content = "Use the content in the %s file, answer the question: 'What is the development of price of aluminium in the near future'. Keep the answer consise within 2 paragraphs. In addition, rate the confidence of the answer on the scale from 0 to 10. Return the result in json format" %concat_file,
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

print('Answer:')

result = messages.data[0].content[0].text.value

print(result)

answer_file = open("answer.txt", "w")
answer_file.write(result)
answer_file.close()
