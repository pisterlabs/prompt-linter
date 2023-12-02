from langchain.llms import OpenAIChat
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.api import open_meteo_docs
import openai
openai.api_key = ""
llm = OpenAIChat(model_name="gpt-3.5-turbo")
chain_new = APIChain.from_llm_and_api_docs(llm,open_meteo_docs.OPEN_METEO_DOCS,verbose=True)

#我们向ChatGPT询问上海当前的温度
chain_new.run('上海现在的温度是多少摄氏度？')