from generative_agent.GenerativeAgent import GenerativeAgent
from datetime import datetime
import guidance
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# guidance.llm = guidance.llms.OpenAI("text-davinci-003", api_key=OPENAI_API_KEY)
# prompt = guidance('''The best thing about the beach is {{~gen 'best' temperature=0.7 max_tokens=7}}''')
# res = prompt()
# print(res)

# exit()

guidance.llm = guidance.llms.OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

now = datetime.now()
new_time = now.replace(hour=7, minute=25)
description = "bocchi is student of shuka high school;She likes to play guitar;She has youtube channel name of guitar hero"
sam = GenerativeAgent(
    guidance=guidance,
    name="Bocchi",
    age=17,
    description=description,
    traits="like to be alone, no speech, like to play guitar",
    embeddings_model=embeddings_model,
    current_time=new_time,
    verbose=True
)

#sam.update_status()
sam.make_plan()

print(sam.plan)
print(sam.get_summary(force_refresh=True))
