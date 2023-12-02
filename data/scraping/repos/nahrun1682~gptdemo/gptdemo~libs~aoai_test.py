#reference:https://python.langchain.com/docs/integrations/llms/azure_openai
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

#参考：https://dev.classmethod.jp/articles/python-langchain-azure-openai-service-model-sample/
from langchain.llms import AzureOpenAI

# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

#os.environで.envファイルにある環境変数を取得
aoai_api_type = os.environ["AOAI_API_TYPE"]
aoai_api_version = os.environ["AOAI_API_VERSION"]
aoai_api_base_url = os.environ["AOAI_API_BASE_URL"]
aoai_api_key = os.environ["AOAI_API_KEY"]
aoai_deployment_name = os.environ["AOAI_DEPLOYMENT_NAME"]


def generate_response_aoai(input_text):
    llm = AzureChatOpenAI(openai_api_key=aoai_api_key,
                        openai_api_type=aoai_api_type,
                        openai_api_base=aoai_api_base_url,
                        openai_api_version=aoai_api_version,
                        deployment_name=aoai_deployment_name,
                        temperature=0)


    messages = [SystemMessage(content='You are a helpful and great assistant.')]
    
    messages.append(HumanMessage(content=input_text))
    res = llm(messages=messages)
    answer = res.content
    messages.append(AIMessage(content=answer))
    return answer


#以下テスト用
# prompt = ''
# while (True):
#     prompt = input('Q: ')

#     if prompt == 'q' or not prompt:
#         break

#     messages.append(HumanMessage(content=prompt))
#     res = llm(messages=messages)
#     answer = res.content
#     messages.append(AIMessage(content=answer))

#     print('A:', answer)

# print('-- Chat History --')
# for msg in messages:
#     print(msg.type, '\t>> ', msg.content)
    



