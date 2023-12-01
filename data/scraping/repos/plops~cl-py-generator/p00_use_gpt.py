#|default_exp p00_use_gpt
# python -m venv ~/llm_env; . ~/llm_env/bin/activate; pip install langchain
# 
# deactivate
import os
import time
import langchain.chat_models
import langchain.schema
import langchain.llms
import openai
start_time=time.time()
debug=True
_code_git_version="2794fadc3cf1cbc00e3f6e1443a304d857940421"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/117_langchain_azure_openai/source/"
_code_generation_time="18:44:19 of Tuesday, 2023-09-12 (GMT+1)"
chatgpt_deployment_name="gpt-35"
chatgpt_model_name="gpt-35-turbo"
openai.api_type="azure"
openai.api_key=os.getenv("OPENAI_API_KEY")
openai.api_base=os.getenv("OPENAI_API_BASE")
openai.api_version=os.getenv("OPENAI_API_VERSION")
# this works
chat=langchain.chat_models.AzureChatOpenAI(deployment_name=chatgpt_deployment_name, model_name=chatgpt_model_name, temperature=1)
user_input=input("Ask me a question: ")
messages=[langchain.schema.HumanMessage(content=user_input)]
print(chat(messages).content)