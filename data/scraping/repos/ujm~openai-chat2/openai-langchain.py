from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage
import time
import pickle

SLEEP_INTERVAL = 5
PICKLE_FILE = 'my_array.pickle'
PROMPT = 'あなた: '
CONTINUE_PROMPT = '>>>'
ERROR_MESSAGE = '処理中...'
EXIT_MESSAGE = '[終了するには "ctrl+c" と入力してください。]\n\n'

print(EXIT_MESSAGE)

# LangChainとOpenAIの初期化
llm = OpenAI()
chat_model = ChatOpenAI()

# プロンプトテンプレートの作成
system_template = "You are a helpful assistant."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# ユーザーからのメッセージを受け取る
input_str = ""  # 初期化

while True:
    input_str = ""

    while True:
        line = input(PROMPT if input_str == "" else CONTINUE_PROMPT)
        if line == "":
            break
        input_str += line + "\n"

    # 最後の改行を削除
    input_str = input_str.rstrip('\n')

    while True:
        try:
            messages = [HumanMessage(content=input_str)]
            response = chat_model.predict_messages(messages)
            break
        except Exception:
            print(ERROR_MESSAGE)
            time.sleep(SLEEP_INTERVAL)

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(input_str, f)

    print(f"アシスタント: {response.content}\n")
    input_str = response
