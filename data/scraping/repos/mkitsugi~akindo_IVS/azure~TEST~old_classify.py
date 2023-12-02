import os

import gradio as gr
import openai
import requests
from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

os.environ["OPENAI_API_KEY"] = "your openai api key"

USER_QUESTION_TYPE_1 = "アウトドアについての質問"
USER_QUESTION_TYPE_2 = "アウトドア商品に関する質問"

AKS_QUESTION_TYPE_TEMPLATE = """
次の発言は以下のどのカテゴリに当てはまる質問か教えて下さい。
1. 恋人を探したい
2. あなた自身への質問
3. 雑談やその他の話題

質問: {question}
"""

ASK_ITEM_NAME_TEMPLATE = """次の質問はどのアウトドア製品に関するものか製品名で教えて下さい。

質問: {question}
製品名:
"""

SUMMARIZE_API_RESPONSE_TEMPLATE = """あなたはアウトドアの専門家です。
あなたは、「{item}」のレビューについてユーザーに質問され、以下のJSONの情報を持っています。
以下のJSONには、「{item}」のレビューが含まれています。
レビュー箇所を特定し、以下の条件に従ってユーザーに対する回答文を作成してください。

・中学生にわかるような言葉で説明してください。
・「です」「ます」で答えてください。
・JSONに関する情報は、除いてください。
・最後にJSONから取得したURLを「参考にしたレビューのURLはこちらです:」の形式で追記してください。

```
{json}
```
"""

template = """
The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.
You are an AI assistant who will be the user's love counselor. 
You mainly use japanese when you reply to the user's questions.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferWindowMemory(k=2, return_messages=True)
# conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
conversation = ConversationChain(llm=llm, memory=memory, verbose=True, prompt=prompt)


def judge_question_type(message: str) -> str:
    prompt = PromptTemplate(
        input_variables=["question"],
        template=AKS_QUESTION_TYPE_TEMPLATE,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(question=message).strip()


def get_item_name(message: str) -> str:
    prompt = PromptTemplate(
        input_variables=["question"],
        template=ASK_ITEM_NAME_TEMPLATE,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(question=message).strip()


def get_item_review(message: str, item: str) -> str:
    url = "https://api.explaza.jp/openai/search"
    headers = {"Content-Type": "application/json"}
    params = {"query": item, "num_results": 3}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception
    response = response.json()
    if len(response["results"]) == 0:
        return conversation.predict(input=message)
    prompt = PromptTemplate(
        input_variables=["json", "item"],
        template=SUMMARIZE_API_RESPONSE_TEMPLATE,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(json=response, item=item).strip()


def chat(message, history):
    history = history or []
    judged = judge_question_type(message)

    answer = ""
    if USER_QUESTION_TYPE_1 in judged:
        answer = conversation.predict(input=message)
    if USER_QUESTION_TYPE_2 in judged:
        item_name = get_item_name(message)
        answer = get_item_review(message, item_name)
    history.append((message, answer))
    return history, history


demo = gr.Interface(
    fn=chat,
    inputs=[gr.Textbox(lines=10, placeholder="Message here..."), "state"],
    outputs=[gr.Chatbot(), "state"],
    theme=gr.themes.Soft(
        primary_hue="emerald",
    ),
    allow_flagging="never",
)
# demo.launch(inline=False, share=False)
demo.launch()
