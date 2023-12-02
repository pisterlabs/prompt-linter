import json
import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import messages_to_dict
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
import base64
import pickle
from langchain.memory import ConversationTokenBufferMemory
from django.shortcuts import render
from rest_framework.views import APIView

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from common.utils import message_response, make_questions

from django.contrib.auth.models import User
from chat.models import BodyText

from django.core.exceptions import ObjectDoesNotExist

# .env ファイルをロードして環境変数へ反映
from dotenv import load_dotenv
load_dotenv()


@require_GET
def hello(request):
    name = request.GET.get('name', '')

    return JsonResponse({'message': "hello"})


@require_GET
def talkQuestion(request):
    message = request.GET.get('message', '')
    print(message)
    response = message_response(message)
    questions = """
{
    "question" : ["返答文1","返答文2","返答文3"]
}
"""
    questions = make_questions(response)

    retdata = {
        "message": response,
        "question_json": questions
    }

    return JsonResponse(retdata)


class Memory(APIView):
    def get(self, request):
        print("get")
        message = request.GET.get('message')
        print(message)

        # ChatOpenAIクラスのインスタンスを作成、temperatureは0.7を指定
        chat = ChatOpenAI(temperature=0.7)
        llm = ChatOpenAI(
            openai_api_base=os.getenv('OPENAI_API_BASE'),
            openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
            model_name='gpt-4', model_kwargs={"deployment_id": "Azure-GPT-4-8k"}, temperature=0)

        # 会話の履歴を保持するためのオブジェクト
        memory = ConversationBufferWindowMemory(return_messages=True, k=3)
        # もとmemory.json

        # 特定のIDのモデルを取得します。

        try:
            instance = BodyText.objects.get(id=1)  # 1はサンプルのIDです

            saved_memory = json.loads(instance.body)
            for i in range(0, len(saved_memory), 2):
                human_item = saved_memory[i]
                ai_item = saved_memory[i+1]
                # print('Human input:', human_item['data']['content'])
                # print('AI output:', ai_item['data']['content'])

                memory.save_context(
                    {'input': human_item['data']['content']},
                    {'output': ai_item['data']['content']})

        except ObjectDoesNotExist:
            instance = BodyText()  # 該当のIDが存在しない場合、新たなインスタンスを作成します。
        except json.decoder.JSONDecodeError as e:
            # JSON文字列が正しくデコードできない場合の例外処理
            print(f"JSONDecodeError: {e}")

        print(memory.load_memory_variables({}))

        # テンプレートを定義
        template = """
        以下は、人間とAIのフレンドリーな会話です。
        AIはその文脈から具体的な内容をたくさん教えてくれます。
        AIは質問の答えを知らない場合、正直に「知らない」と答えます。
        """

        # テンプレートを用いてプロンプトを作成
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # AIとの会話を管理
        conversation = ConversationChain(
            llm=chat, memory=memory, prompt=prompt)

        # ユーザからの入力を受け取り、変数commandに代入
        command = message
        response = conversation.predict(input=command)
        print(response)
        print(dir(response))

        # memoryオブジェクトから会話の履歴を取得して、変数historyに代入
        history = memory.chat_memory

        # 会話履歴をJSON形式の文字列に変換、整形して変数messagesに代入
        messages = json.dumps(messages_to_dict(
            history.messages), indent=2, ensure_ascii=False)

        instance.body = messages  # モデルのフィールドに値を代入します。

        print("savesavesave")
        print(messages)
        instance.save()  # データベースに保存します。

        '''

        with open("data.txt", "w", encoding='utf-8') as outfile:
            outfile.write(messages)
        '''
        return JsonResponse({"message": response})
