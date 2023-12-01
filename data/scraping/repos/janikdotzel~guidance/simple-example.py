import guidance
from dotenv import load_dotenv

load_dotenv()

guidance.llms.OpenAI.cache.clear()
guidance.llm = guidance.llms.OpenAI("text-davinci-003")
prompt = guidance('''The best thing about the beach is {{~gen 'best' temperature=0.7 max_tokens=7}}''')
prompt = prompt()
print(prompt)
