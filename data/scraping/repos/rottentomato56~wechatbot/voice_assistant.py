import os
import requests
import settings
import time
import re
import ffmpeg
import openai
import wechat
from pydub import AudioSegment
from hanziconv import HanziConv

openai.api_key = settings.OPENAI_API_KEY

# play.ht voice models for chinese
MODELS = [
    'zh-CN_LiNaVoice',
    'zh-CN_ZhangJingVoice',
    'zh-CN-XiaoxiaoNeural',
    'zh-CN-XiaoyouNeural',
    'zh-CN-HuihuiRUS',
    'zh-CN-Yaoyao-Apollo',
    'zh-CN-XiaohanNeural',
    'zh-CN-XiaomoNeural',
    'zh-CN-XiaoruiNeural',
    'zh-CN-XiaoxuanNeural',
    'zh-CN-XiaoshuangNeural'
]

CHINESE_MODEL = 'zh-CN-XiaomoNeural'

CONVERSION_URL = 'https://play.ht/api/v1/convert'

def text_to_speech(message, model=CHINESE_MODEL):
    message = prepare_text(message)

    payload = {
        'content': [message],
        'voice': model
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "AUTHORIZATION": settings.VOICE_AI_API_KEY,
        "X-USER-ID": settings.VOICE_AI_USER_ID
    }

    response = requests.post(CONVERSION_URL, json=payload, headers=headers)

    transcription_id = response.json().get('transcriptionId')

    # poll for job success, eventually migrate this to webhook

    job_done = False
    url = f"https://play.ht/api/v1/articleStatus?transcriptionId={transcription_id}"

    headers = {
        "accept": "application/json",
        "AUTHORIZATION": settings.VOICE_AI_API_KEY,
        "X-USER-ID": settings.VOICE_AI_USER_ID
    }

    while not job_done:
        response = requests.get(url, headers=headers)
        if response.json().get('converted'):
            job_done = True
            audio_file = response.json().get('audioUrl')
            audio_duration = response.json().get('audioDuration')
        else:
            time.sleep(2)

    response = requests.get(audio_file)

    filename = transcription_id + '.mp3'
    with open(filename, 'wb') as f:
        f.write(response.content)

    if audio_duration > 60:
        trimmed_filename = transcription_id.replace('-', '') + '_trimmed.mp3'
        stream = ffmpeg.input(filename)
        stream = ffmpeg.output(stream, trimmed_filename, t=59)
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)
        os.remove(filename)
        return trimmed_filename

    return filename


def has_english(text):
    """ 
    Will return True or False depending on if the text contains more than 8 english words. 
    Use this condition to determine if it is necessary to convert the text to speech  
    """
    english_words = re.findall(r'\b[A-Za-z\-]+\b', text)
    return len(english_words) > 8

def prepare_text(text):
    english_sections = re.findall(r'\b[A-Za-z\s.,;!?\-]+\b', text)
    for section in english_sections:
        text = text.replace(section, f',{section},', 1)
    return text

def test():
    s = '你可以学习这个短语 "self-care"（自我关怀）来描述一个人照顾自己身心健康的行为和习惯。例如，你可以说 "Practicing self-care is important for maintaining a healthy lifestyle."（实施自我关怀对于保持健康的生活方式很重要）。这个短语可以帮助你学习如何照顾自己的身心健康，与"laughter is the best medicine" 相关。你可以学习这个短语 "self-care"（自我关怀）来描述一个人照顾自己身心健康的行为和习惯。例如，你可以说 "Practicing self-care is important for maintaining a healthy lifestyle."（实施自我关怀对于保持健康的生活方式很重要）。这个短语可以帮助你学习如何照顾自己的身心健康，与"laughter is the best medicine" 相关。你可以学习这个短语 "self-care"（自我关怀）来描述一个人照顾自己身心健康的行为和习惯。例如，你可以说 "Practicing self-care is important for maintaining a healthy lifestyle."（实施自我关怀对于保持健康的生活方式很重要）。这个短语可以帮助你学习如何照顾自己的身心健康，与"laughter is the best medicine" 相关。'
    return text_to_speech(s)


def get_voice_message(media_id):
    file_id = str(media_id).replace('-', '')
    file_id = 'sample'
    access_token = wechat.get_access_token()
    url = f'https://api.weixin.qq.com/cgi-bin/media/get?access_token={access_token}&media_id={media_id}'
    response = requests.get(url)
    amr_in = file_id + '.amr'
    mp3_out = file_id + '.mp3'
    with open(amr_in, 'wb') as f:
        f.write(response.content)

    amr_audio = AudioSegment.from_file(amr_in, format='amr')
    mp3_audio = amr_audio.export(mp3_out, format='mp3')
    transcript = openai.Audio.transcribe('whisper-1', mp3_audio)
    text = transcript.get('text')
    return text


def transcribe_audio(amr_in):
    file_id = os.path.basename(amr_in).replace('.amr', '')
    mp3_out = file_id + '.mp3'

    amr_audio = AudioSegment.from_file(amr_in, format='amr')
    mp3_audio = amr_audio.export(mp3_out, format='mp3')
    transcript = openai.Audio.transcribe('whisper-1', mp3_audio)
    text = transcript.get('text')
    os.remove(amr_in)
    os.remove(mp3_out)
    return text



