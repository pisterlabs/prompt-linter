from django.shortcuts import render
from django.http import HttpResponse
import openai
import os, json
from pathlib import Path
from googletrans import Translator
from django.core.exceptions import ImproperlyConfigured
import my_settings


# def translate_text(text, target_language):
#     translator = Translator()
#     result = translator.translate(text, dest=target_language)
#     translated_text = result.text
#     return translated_text

# 번역할 텍스트

# BASE_DIR = Path(__file__).resolve().parent.parent

# secret_file = os.path.join(BASE_DIR, 'secrets.json') # secrets.json 파일 위치를 명시

# with open('secrets.json') as f:
#     secrets = json.loads(f.read())


openai.organization =  my_settings.GPT_ORGANIZATION_ID
openai.api_key = my_settings.GPT_SECRET_KEY

# def get_secret(setting, secrets=secrets):
#     try:
#         return secrets[setting]
#     except KeyError:
#         error_msg = "Set the {} environment variable".format(setting)
#         raise ImproperlyConfigured(error_msg)



def chat_with_gpt_summary(prompt):
    # openai.organization =  get_secret("GPT_ORGANIZATION_ID")
    # openai.api_key = get_secret("GPT_SECRET_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # ChatGPT 모델 선택
        messages=[
            {"role": "system", "content": "Please write in Korean language."},
            {"role": "user", "content": prompt},
            {"role": "system", "content": "한문장으로 요약해줘"}]
    )
    reply = response["choices"][0]["message"]["content"]
    return reply

def chat_with_gpt_correction(prompt):
    # openai.organization =  get_secret("GPT_ORGANIZATION_ID")
    # openai.api_key = get_secret("GPT_SECRET_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # ChatGPT 모델 선택
        messages=[
            {"role": "system", "content": "Please write in Korean language."},
            {"role": "user", "content": prompt},
            {"role": "system", "content": "글을 첨삭해줘."}]
    )
    reply = response["choices"][0]["message"]["content"]
    return reply

def chat_with_gpt_creation(prompt):
    # openai.organization =  get_secret("GPT_ORGANIZATION_ID")
    # openai.api_key = get_secret("GPT_SECRET_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # ChatGPT 모델 선택
        messages=[
            {"role": "system", "content": "Please write in Korean language."},
            {"role": "user", "content": prompt},
            {"role": "system", "content": "글을 이어서 완성해줘."}]
    )
    reply = response["choices"][0]["message"]["content"]
    return reply

    # text_to_translate = "번역기능 성공"
    # # 번역 실행
    # translated_text = translate_text(text_to_translate, 'en')
def index(request):
    if request.method == 'GET':
        return render(request, "index.html")
    else:
        chat_prompt = request.POST.get('contents', None)
        option = request.POST.get('write_category', None)
        if option == 'summary':
            response = chat_with_gpt_summary(chat_prompt)
        elif option == 'creation':
            response = chat_with_gpt_creation(chat_prompt)
        else:
            response = chat_with_gpt_correction(chat_prompt)
        return render(request, "index.html", {'response':response})

