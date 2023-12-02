import openai
import environ

env = environ.Env()
environ.Env.read_env(env_file='../../.env')

print(env('OPENAI_API_KEY'))

openai.api_key = env('OPENAI_API_KEY')
openai.File.create(
  file=open("files/question_chat_find_tuning.jsonl", "rb"),
  purpose='fine-tune'
)

openai.FineTuningJob.create(training_file="file-abc123", model="gpt-3.5-turbo")