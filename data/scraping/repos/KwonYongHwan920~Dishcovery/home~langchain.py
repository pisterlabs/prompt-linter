from dotenv import load_dotenv
import os
from django.http import JsonResponse
import requests, json
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import OpenAI
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

@method_decorator(csrf_exempt, name='dispatch')
def food_recommendation(request):
    if (request.method == 'GET'):
        user_input = request.GET.get('user_input')
    
        prompt = PromptTemplate(
            user_input = ['user_input'],
            template= "{user_input}을 좋아하는 사람에게 추천할 만한 음식을 추천해줘."
        )
    
        llm = OpenAI(temperature=0.8)
    
        user_input = user_input.replace(" ", "")
        user_input = user_input.replace("\n", "")
        user_input = user_input.replace("\t", "")
        user_input = user_input.replace("\'", "")
        user_input = user_input.replace("\"", "")
        user_input = user_input.replace("\r", "")
    
        chain = LLMChain(llm=llm, prompt=prompt)
    
        chain.run("호불호 없는 맛있는 음식")
    
        result = (prompt.format(chain.run))
    
        if not user_input:
            return JsonResponse({'message': 'NO_KEY'}, status=400)
    
        if user_input == '종료':
            return JsonResponse({'message': 'SUCCESS', 'result': '종료'}, status=200)
    
        return JsonResponse({'message': 'SUCCESS', 'result': result}, status=200)
    
    else:
        return JsonResponse({'message': 'INVALID_HTTP_METHOD'}, status=405)


@method_decorator(csrf_exempt, name='dispatch')
def food_assistace(request):
    if (request.method == 'GET'):
        llm = OpenAI(temperature=0)
        conversation = ConversationChain(llm=llm, verbose=True)
    
        user_input = request.GET.get('user_input')
    
        while True:
            output = conversation.predict(user_input)
            return JsonResponse({'message': 'SUCCESS', 'result': output}, status=200)

            if user_input == '종료':
                break
            
    else:
        return JsonResponse({'message': 'INVALID_HTTP_METHOD'}, status=405)
