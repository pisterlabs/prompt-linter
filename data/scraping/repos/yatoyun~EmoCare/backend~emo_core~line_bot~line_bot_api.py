from google.cloud import language_v1
from ..models import UserModel, TempRegisterToken
from emo_core.models import EmotionData, ChatLogs, AdviceData
from django.http import HttpResponse, HttpResponseForbidden, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from pathlib import Path
import uuid
import environ
import openai
import random
import string
from django.utils import timezone
from datetime import timedelta

# Environment and Configuration
BASE_DIR = Path(__file__).resolve().parent.parent

# import env file
env = environ.Env(DEBUG=(bool, False))
environ.Env.read_env(Path(BASE_DIR, '.env'))
ACCESS_TOKEN = env('LINE_BOT_ACCESS_TOKEN')
CHANNEL_SECRET = env('LINE_BOT_CHANNEL_SECRET')
GOOGLE_APPLICATION_CREDENTIALS = env('GOOGLE_APPLICATION_CREDENTIALS')
GPT_API_KEY = env('GPT_API_KEY')
REGISTER_URL = env('REGISTER_URL')

# Initialize APIs
line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

def generate_unique_token():
    return str(uuid.uuid4())

@csrf_exempt
def callback(request):
    if request.method == 'POST':
        signature = request.META['HTTP_X_LINE_SIGNATURE']

        try:
            handler.handle(request.body.decode('utf-8'), signature)
        except (InvalidSignatureError, LineBotApiError):
            return HttpResponseForbidden()
        return HttpResponse()
    return HttpResponseBadRequest()


def send_temporary_code(line_user_id):
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # LINEユーザーに送信
    line_bot_api.push_message(
        line_user_id,
        TextSendMessage(text=f"Your temporary code:")
    )

    line_bot_api.push_message(
        line_user_id,
        TextSendMessage(text=str(code))
    )

    return code

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_user_id = event.source.user_id
    text = event.message.text.lower()  # Normalize text

    if text == 'register':
        handle_registration(event, line_user_id)
    else:
        handle_chat(event, line_user_id, text)
    
# register functions
def handle_registration(event, line_user_id):
    reply_text = ""
    try:
        with transaction.atomic():
            user, created = UserModel.objects.get_or_create(line_user_id=line_user_id)
            if created:
                unique_token = generate_unique_token()
                expiration_time = timezone.now() + timedelta(minutes=3)
                if TempRegisterToken.objects.filter(line_user_id=line_user_id).exists():
                    TempRegisterToken.objects.filter(line_user_id=line_user_id).delete()
                TempRegisterToken.objects.create(line_user_id=line_user_id, token=unique_token, expiration=expiration_time)
                registration_url = f"{REGISTER_URL}?token={unique_token}"  # Ensure this is HTTPS
                reply_text = "Please complete your registration by visiting this link: " + registration_url
            else:
                reply_text = "You are already registered"
    except Exception as e:
        # Log the exception for debugging purposes
        reply_text = "An error occurred. Please try again later."
        print(e)

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

# chat functions
def handle_chat(event, line_user_id, text):
    try:
        django_user = UserModel.objects.get(line_user_id=line_user_id)
    except UserModel.DoesNotExist:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="You are not registered."))
        return

    # analyze and reply
    emotion_score, emotion_magnitude = analyze_emotion(text)
    advice = generate_advice(text, emotion_score)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=advice))
    
    # save data
    ChatLogs.objects.create(user=django_user, message=text)
    EmotionData.objects.create(user=django_user, emotion_score=emotion_score, emotion_magnitude=emotion_magnitude)
    AdviceData.objects.create(user=django_user, advice=advice)
    
def analyze_emotion(content):
    client = language_v1.LanguageServiceClient.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    document = language_v1.Document(content=content, type_=language_v1.Document.Type.PLAIN_TEXT, language='ja')
    response = client.analyze_sentiment(document=document)
    return response.document_sentiment.score, response.document_sentiment.magnitude

def load_chat_logs():
    # new 5 logs
    chat_logs = list(ChatLogs.objects.all().order_by('-created_at')[:5])
    advice_logs = list(AdviceData.objects.all().order_by('-created_at')[:5])
    
    messages = []
    # format chat logs
    for chat, advice in zip(reversed(chat_logs), reversed(advice_logs)):
        messages.append({"role": "user", "content": chat.message})
        messages.append({"role": "assistant", "content": advice.advice})
        
    return messages

def load_emotion_logs(new_emotion_score):
    # new 5 logs
    emotion_logs = EmotionData.objects.all().order_by('-created_at')[:4]
    
    avg_score = 0
    
    for emotion in emotion_logs:
        avg_score += emotion.emotion_score
    
    avg_score += new_emotion_score
    if len(emotion_logs) > 0:
        avg_score /= len(emotion_logs) + 1
    
    return round(avg_score, 2)

def generate_advice(text, new_emotion_score):
    openai.api_key = GPT_API_KEY
    openai.Model.list()
    # parameters for GPT-3.5
    system_prompt = "あなたは相手の心を気遣うことが上手く、メンタルケアが得意です。相手のmessageとその感情分析値が渡されます。その感情分析値も考慮して返信してください。相手と同じ長さの返信をしてください。50文字以内で"
    max_tokens = 200
    
    # load chat logs
    old_messages = load_chat_logs()
    emotion_score = load_emotion_logs(new_emotion_score)  
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(old_messages)
    messages.append({"role": "user", "content": f"{text}:{emotion_score}"})
    
    # generate advice using GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        max_tokens=max_tokens,
        n=1,
        temperature=0.7,
    )

    # extract advice from GPT-3.5 response
    return response['choices'][0]['message']['content'].strip()