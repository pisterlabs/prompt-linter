from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 
client = OpenAI()

# client.files.create(
#   file=open("output.jsonl", "rb"),
#   purpose="fine-tune"
# )
# print(client.files.list())

# client.fine_tuning.jobs.create(
#   training_file="file-Lbk5nm03pOrfUi2prkaely10", 
#   model="gpt-3.5-turbo"
# )

# print(client.fine_tuning.jobs.list())
# print(client.fine_tuning.jobs.retrieve("ftjob-0VMns7KgPfzxnvYIFUAXYKRc"))

# print(client.files.retrieve_content("file-SDnfYFQORCkd4AKTHKSyPRxm"))


