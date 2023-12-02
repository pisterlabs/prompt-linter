from langchain import OpenAI, LLMChain, PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
model_instance = OpenAI(openai_api_key=api_key,
                        model="text-davinci-002", max_tokens=2048, streaming=True)

plan_template = "Create me a custom study plan for {theme}. Divide it to {days} days. I am a {level} student."
plan_prompt = PromptTemplate(
    input_variables=["theme", "days", "level"], template=plan_template)

llm_plan = LLMChain(
    llm=model_instance,
    prompt=plan_prompt,
    verbose=True
)
