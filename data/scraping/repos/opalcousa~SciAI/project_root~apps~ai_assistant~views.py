from django.shortcuts import render
from django.http import JsonResponse
from .models import AIAssistant
from openai import OpenAI, GPT4

# Initialize OpenAI's GPT-4 API
openai = OpenAI(api_key='your-api-key')
gpt4 = GPT4(openai)

def chat(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        response = gpt4.chat(message)
        return JsonResponse({'response': response})
    else:
        return render(request, 'ai_assistant/chat.html')

def get_assistant_info(request, assistant_id):
    assistant = AIAssistant.objects.get(id=assistant_id)
    return JsonResponse({'assistant': assistant.to_dict()})

def update_assistant_info(request, assistant_id):
    if request.method == 'POST':
        assistant = AIAssistant.objects.get(id=assistant_id)
        assistant.update(request.POST)
        return JsonResponse({'status': 'success'})
    else:
        return JsonResponse({'status': 'failed'})

def delete_assistant(request, assistant_id):
    if request.method == 'POST':
        assistant = AIAssistant.objects.get(id=assistant_id)
        assistant.delete()
        return JsonResponse({'status': 'success'})
    else:
        return JsonResponse({'status': 'failed'})