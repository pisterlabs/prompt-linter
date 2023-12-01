import os

from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from database import db_session, init_db, schema
from dotenv import load_dotenv

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_APIKEY")

prompt_template_name = PromptTemplate(
    input_variables=["name"],
    template="How do I make an HTTP request in {name}?, Generate code for it",
)

repo_id = "tiiuae/falcon-40b-instruct"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 1000})

chain = LLMChain(llm=llm, prompt=prompt_template_name)  
response = chain.run(name="JavaScript")

print(f"{response}", "response")

