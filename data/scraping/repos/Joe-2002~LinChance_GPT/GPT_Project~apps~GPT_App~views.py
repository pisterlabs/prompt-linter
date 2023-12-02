from rest_framework.decorators import action
from rest_framework.response import Response
# from django.http import JsonResponse
from .models import *
from .serializers import *
from rest_framework.viewsets import ModelViewSet
from apps.GPT_API.gpt import get_completion
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import os
from datetime import datetime
from rest_framework import status
from apps.GPT_API.spark import SparkLLM
from apps.Lin_FAISS.MyFaiss import *
from apps.Prompts.lin_prompt import *
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import *
from apps.GPT_API.configs import openai_api_key,serpapi_api_key
from apps.Prompts.lin_prompt import *
from langchain import PromptTemplate
# Create your views here.



os.environ['OPENAI_API_KEY'] = openai_api_key  
os.environ["SERPAPI_API_KEY"] = serpapi_api_key

### GPT3.5

class ChatHistoryViewSet(ModelViewSet):
    queryset = ChatHistory.objects.all()
    serializer_class = ChatHistorySerializer

## GPT3.5
    @action(detail=False, methods=['post'])
    def chat_with_gpt(self, request):
        data = request.data['postdata']
        print("这是前端传过来的数据",data)
        user_input = data.get('user_input')  # 获取用户输入
        chat_name = data.get('chat_name')  # 获取对话名称
        new_chat = data.get('new_chat')  # 是否新建对话，假设 'new_chat' 是布尔字段
        # print (new_chat)
        
        if new_chat == 'True':
            # 如果新建对话，创建 ChatHistory 记录，并初始化一个空的历史记录列表
            time = datetime.now()
            time = str(time)
            # print(time)
            chat_history = ChatHistory.objects.create(timestamp=time, chat_name=chat_name)

        elif new_chat == 'False':
            # 如果不是新建对话，查找最近的 ChatHistory 记录，并构建历史记录列表
            print('*********************')
            chat_history = ChatHistory.objects.latest('timestamp')
            print(chat_history)


        prompt = f"Start the conversation:\n"
        for message in chat_history.messages.all():
            if message.user_input:
                prompt += f"User: {message.user_input}\n"
            if message.gpt_response:
                prompt += f"GPT: {message.gpt_response}\n"
        prompt += f"User: {user_input}\n"


        # 调用 GPT-3.5获取 AI 回复
        gpt_response = get_completion(prompt)
        print(gpt_response)
        print('------------------------------------------------------')

        # 创建新的 ChatMessage 记录并关联到 ChatHistory，保存
        chat_message = ChatMessage.objects.create(chat_history=chat_history, user_input=user_input, gpt_response=gpt_response)
        chat_history.messages.add(chat_message)

        # 返回 AI 回复
        return Response({'gpt': gpt_response})

### Spark

    @action(detail=False, methods=['post'])
    def chat_with_Spark(self, request):
        data = request.data['postdata']
        print("这是前端传过来的数据",data)
        user_input = data.get('user_input')  # 获取用户输入
        chat_name = data.get('chat_name')  # 获取对话名称
        new_chat = data.get('new_chat')  # 是否新建对话，假设 'new_chat' 是布尔字段
        # print (new_chat)
        print(user_input,chat_name,new_chat)

        if new_chat == "True":
            # 如果新建对话，创建 ChatHistory 记录，并初始化一个空的历史记录列表
            time = datetime.now()
            time = str(time)
            # print(time)
            chat_history = ChatHistory.objects.create(timestamp=time, chat_name=chat_name)
            history = []
        if new_chat == "False":
            # 如果不是新建对话，查找最近的 ChatHistory 记录，并构建历史记录列表
            print('*********************')
            chat_history = ChatHistory.objects.latest('timestamp')
            print(chat_history)
            history = []

        prompt = f"Start the conversation:\n"
        for message in chat_history.messages.all():
            if message.user_input:
                prompt += f"User: {message.user_input}\n"
            if message.gpt_response:
                prompt += f"Spark: {message.gpt_response}\n"
        prompt += f"User: {user_input}\n"


        # 调用 SparkLLM 函数获取 AI 回复
        gpt_response, updated_history = SparkLLM(prompt, history)
        print(gpt_response)
        print('------------------------------------------------------')

        # 创建新的 ChatMessage 记录并关联到 ChatHistory，保存
        chat_message = ChatMessage.objects.create(chat_history=chat_history, user_input=user_input, gpt_response=gpt_response)
        chat_history.messages.add(chat_message)

        # 返回 AI 回复
        return Response({ gpt_response})

### Chat History


    @action(detail=False, methods=['get', 'post'])
    def chat_history(self, request):
        if request.method == 'GET':
            all_timestamps = ChatHistory.objects.values_list('timestamp', flat=True)
            # all_chat_name = ChatHistory.objects.values_list('chat_name',flat=True)
            return Response(all_timestamps)

        
        if request.method == 'POST':
            selected_timestamp = request.data.get('timestamp')
            chat_name = request.data.get('chat_name')

            # 获取匹配的 ChatHistory 记录
            chat_history = ChatHistory.objects.filter(timestamp=selected_timestamp).first()

            if chat_history:
                # 如果传入了 chat_name，则更新 chat_history 的 chat_name
                if chat_name is not None:
                    chat_history.chat_name = chat_name
                    chat_history.save()

                serializer = ChatHistorySerializer(chat_history)
                return Response(serializer.data)
            else:
                return Response({"message": "No chat history found for the selected date."}
                                )


### Faiss


    @action(detail=False, methods=['post'])
    def chat_with_Faiss(self, request):

        data = request.data['postdata']
        print(data)
        user_input = data.get('user_input')  # 获取用户输入
        print('-----------------------------------')
        print('user_input:',user_input)
        chat_name = data.get('chat_name')  # 获取对话名称
        print('-----------------------------------')
        print('chat_name:',chat_name)
        new_chat = data.get('new_chat')  # 是否新建对话，假设 'new_chat' 是布尔字段
        print('-----------------------------------')
        print('new_chat:',new_chat)
        query = user_input
        if new_chat == "True":
            # 如果新建对话，创建 ChatHistory 记录，并初始化一个空的历史记录列表
            time = datetime.now()
            time = str(time)
            chat_history = ChatHistory.objects.create(timestamp=time, chat_name=chat_name)
            history = []
        elif new_chat == "False":
            # 如果不是新建对话，查找最近的 ChatHistory 记录，并构建历史记录列表
            chat_history = ChatHistory.objects.latest('timestamp')
            history = []
        linfaiss = LinFaiss()
        # 获取相似内容
        similar_doc = []
        print('------------------------------------------------------')
        similar_doc_all = linfaiss.get_similarity_documents(query, limit=1)
        for data in similar_doc_all:
            similar_doc.append(data['page_content'])
        content, history = SparkLLM(question=query, history=history, similar_doc=similar_doc)

        chat_message = ChatMessage.objects.create(chat_history=chat_history, user_input=user_input, gpt_response=content)
        chat_history.messages.add(chat_message)
        # linfaiss.save_vec_data()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('chat with Faiss')
        print(content)
        print('***********************************************************************')
        return Response({content})



