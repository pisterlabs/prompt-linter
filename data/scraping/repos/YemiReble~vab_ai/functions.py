from django.conf import settings
from django.http import JsonResponse
from pytube import YouTube
import assemblyai as aai
from secret import assemblyal_key, openai_key, cohere_api_key
from secret import token_sid, token_sidcc, token_sidts
from bardapi import Bard, SESSION_HEADERS

import requests
import cohere
import json
import openai
import os


def is_password_up_to_standard(password: str) -> bool:
    """ Implement your checks here
    """
    if len(password) < 8:
        return False
    if not any(char.isdigit() for char in password):
        return False
    if not any(char.isalpha() for char in password):
        return False
    return True


def email_check(email: str) -> bool:
    """ Check email standard
    """
    if email is None:
        return False
    if '@' and '.com' not in email:
        return False

    # Acceptable email provider
    # mails = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outmail.com'}
    # for m in mails:
    #    if m not in email:
    #        return False

    return True


def content_formatter(content: str) -> str:
    """ This fucntion makes sure that the
        content generate from the results
        of transcribing to blog post is well
        formatted in a readable format
    """
    # Split the text into paragraphs based on double line breaks
    paragraph = [para.strip()
                 for para in content.split('\n\n') if para.strip()]

    # Generate HTML content with <p> tags for each paragraph
    html_content = '<div class="max-w-2xl">'
    for para in paragraph:
        html_content += f'<p class="mb-4">{para}</p>'
    html_content += '</div>'

    return html_content


def get_youtube_title(link: str) -> str:
    """ The function that handles the youtube title
        operation
    """
    video = YouTube(link)
    title = video.title
    return title


def get_youtube_audio(link: str):
    """ The function that gets and downloads the audio of the youtube
        link the user provided
    """
    video = YouTube(link)
    audio = video.streams.filter(only_audio=True).first()
    audio_data = audio.download(output_path=settings.MEDIA_ROOT)
    base, extention = os.path.splitext(audio_data)
    downl_file = base + '.mp3'
    os.rename(audio_data, downl_file)
    return downl_file


def get_youtube_transcription(audio_path: str):
    """ The function that gets the transcription of the youtube
        link the user provided
    """
    aai.settings.api_key = assemblyal_key
    audio = get_youtube_audio(audio_path)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio)
    return transcript.text


def generate_blog_from_cohere(transcript: str):
    """ This function uses the Cohere AI to generate
        the required blog post form the user
    """
    try:
        co = cohere.Client(cohere_api_key)
        prompt = f'You are a professional writer. Base on the following \
                transcript from a YouTube video, generate a comprehensive \
                blog article using the following transcript and do not \
                make it look like it is from a \
                YouTube video:\n\n{transcript}\n\nArticle:'

        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=2500,
            temperature=0.9,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        return response.generations[0].text
    except Exception:
        return 'Unable to generate blog content'


def generate_blog_from_openai(transcript: str):
    """ This function uses the OpenAI ChatGPT to generate the
        Intended Blog Article
    """
    try:
        openai.api_key = openai_key

        prompt = f'You are a professional writer. Base on the following \
                transcript from a YouTube video, generate a comprehensive \
                blog article using the following transcript and do not \
                make it look like it is from a \
                YouTube video:\n\n{transcript}\n\nArticle:'

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=1000,
        )

        blog_content = response.choices[0].text.strip()
        return blog_content

    except Exception as e:
        return f'Unable to generate blog content: {e}'


def generate_blog_from_bard(transcript: str):
    """ The function will utilise the Google Bard for free
        to generate the blog content expected by the user's
        operation
    """
    try:
        session = requests.Session()
        token = token_sid
        session.cookies.set("__Secure-1PSID", token_sid)
        session.cookies.set("__Secure-1PSIDCC", token_sidcc)
        session.cookies.set("__Secure-1PSIDTS", token_sidts)
        session.headers = SESSION_HEADERS

        prompt = f'You are a professional writer. Base on the following \
                transcript from a YouTube video, generate a comprehensive \
                blog article using the following transcript and do not \
                make it look like it is from a \
                YouTube video:\n\n{transcript}\n\nArticle:'

        bard = Bard(token=token, session=session)
        blog_content = bard.generate(prompt)['content']
        return blog_content
    except Exception:
        return 'Unable to generate blog content'


def download_youtube_audio(request):
    """ Bonous Function For User To Download The
        Auido File To their Computer
    """

    link = request.GET.get('link')
    audio_file = get_youtube_audio(link)

    with open(audio_file, 'rb') as f:
        response = HttpResponse(f.read(), content_type='audio/mpeg')
        response['Content-Disposition'] = 'attachment; filename={}'
        format(os.path.basename(audio_file))

    return response
