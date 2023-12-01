import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.File.create(
  file=open("./datasets/openai_datasets/bitcoin_chatbot_training_data.jsonl"),
  purpose='fine-tune'
)

# # Examine files and add most recent file id to .env
print(openai.File.list())
openai_file_list = openai.File.list()
new_file = openai_file_list['data'][-1]['id']
print(new_file)

# with open("gpt3/.env", 'a') as outfile:    
#         outfile.write('\n')
#         json.dump('TRAINING_FILE='+new_file, outfile, separators=('"', ''))