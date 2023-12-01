from __future__ import absolute_import, unicode_literals
from celery import shared_task


@shared_task
def separate_audio_from_file(video_id):
    import os
    from yt_dlp import YoutubeDL
    from experiments.models import Video
    from constants import TEMP_LOCAL_PATH, FLOW_STATUS
    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="Start separate_audio_from_file: {}".format(video_id))
    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        return
    video.status = FLOW_STATUS.AUDIO_VIDEO_SEPARATION_IN_PROGRESS
    video.save()

    output_file = TEMP_LOCAL_PATH.TEMP_INPUT_AUDIO.format(video_id)
    if not os.path.isfile(output_file):
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'paths': {'home': output_file.split('/')[0]},
            'outtmpl': {'default': output_file.split('/')[1]},
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }]
        }
        with YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([video.youtube_url])
            if error_code != 0:
                raise Exception('Failed to download video')

        video.status = FLOW_STATUS.AUDIO_VIDEO_SEPARATION_COMPLETED
        video.save()

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="End separate_audio_from_file: {}".format(video_id))
    create_transcription.delay(video_id=video_id)


@shared_task
def create_transcription(video_id, use_openai_for_translating=True):
    from django.conf import settings
    import openai
    import os
    from google.cloud import translate_v2
    from experiments.models import Video
    from constants import TEMP_LOCAL_PATH, FLOW_STATUS, PROMPT_INPUT_LANG_MAPPING, PROMPT_OUTPUT_LANG_MAPPING

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="Start create_transcription: {}".format(video_id))
    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        return

    output_language = str(video.output_language).lower()
    input_language = str(video.input_language).lower()

    if not video.transcription:
        # Updating status
        video.status = FLOW_STATUS.TRANSCRIPTION_IN_PROCESS
        video.save()

        try:
            existing_video = Video.objects.filter(
                youtube_url=video.youtube_url, input_language=input_language
            ).latest('id')
        except Video.DoesNotExist:
            transcription = None
            pass
        else:
            transcription = existing_video.transcription
            print("Transcription exists: {}\n".format(video_id, transcription))

        if not transcription:
            # Getting extracted audio file from temp folder.
            audio_file = open(TEMP_LOCAL_PATH.TEMP_INPUT_AUDIO.format(video.id), "rb")

            # Setting prompt to better result
            prompt = PROMPT_INPUT_LANG_MAPPING.get(input_language, None)
            if not prompt:
                prompt = PROMPT_INPUT_LANG_MAPPING.get('default', None)
            print(prompt)

            # Calling openAI API to transcribe local audio to given input language
            openai.api_key = settings.OPEN_AI_TOKEN
            transcription = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                response_format='text',
                language=input_language,
                temperature=0.3,
                prompt=prompt
            )

        # Updating status
        video.transcription = transcription
        video.save()

    if not video.translated_text and video.transcription:
        if use_openai_for_translating:
            # Setting prompt to better output.
            prompt = PROMPT_OUTPUT_LANG_MAPPING.get((output_language, video.voice_gender), None)
            if not prompt:
                prompt = PROMPT_OUTPUT_LANG_MAPPING.get(("default", video.voice_gender), '').format(output_language)
            prompt += " \n\n {}".format(video.transcription)
            print(prompt)

            # Calling openAI API to translate transcribed text to english/hindi
            openai.api_key = settings.OPEN_AI_TOKEN
            response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt,
                max_tokens=3000,
                temperature=0.2,
                n=1,
                stop=None,
            )
            translated_text = response.get('choices')[0]['text']
        else:
            # For fallback condition we might use Google Translate API.
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS_PATH
            translate_client = translate_v2.Client()
            result = translate_client.translate(video.transcription, target_language=video.output_language)
            translated_text = result.get('translatedText')

        video.status = FLOW_STATUS.TRANSCRIPTION_COMPLETED
        video.translated_text = translated_text
        video.save()

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="End create_transcription: {}".format(video_id))
    convert_text_to_speech.delay(video_id=video_id)


@shared_task
def extract_title(video_id):
    import requests
    import re
    from bs4 import BeautifulSoup
    from experiments.models import Video
    from constants import FLOW_STATUS, YOUTUBE_VIDEO_DURATION_LIMIT

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="Start extract_title: {}".format(video_id))

    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        return
    try:
        response = requests.get(video.youtube_url)
        soup = BeautifulSoup(response.text, "html.parser")
        title_element = soup.find("meta", property="og:title")
        duration_element = soup.find("meta", itemprop="duration")
    except Exception as e:
        send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                                 text="ERROR: {}".format(str(e)))
        return

    if duration_element:
        duration = duration_element["content"]
        duration_pattern = re.compile(r'PT(\d+H)?(\d+M)?(\d+S)?')
        duration_parts = duration_pattern.match(duration).groups()

        hours = int(duration_parts[0][:-1]) if duration_parts[0] else 0
        minutes = int(duration_parts[1][:-1]) if duration_parts[1] else 0
        seconds = int(duration_parts[2][:-1]) if duration_parts[2] else 0

        total_seconds = hours * 3600 + minutes * 60 + seconds
        if total_seconds:
            video.duration = total_seconds

    title = title_element["content"]
    if title:
        video.title = title
        video.slug = video.get_unique_slug()
        video.save()

    if video.duration > YOUTUBE_VIDEO_DURATION_LIMIT:
        video.status = FLOW_STATUS.LONG_LENGTH_ERROR
        video.save()
        send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                                 text="LONG LENGTH ERROR: {}".format(video_id))
        return

    separate_audio_from_file.delay(video_id=video_id)
    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="End extract_title: {}".format(video_id))


@shared_task
def convert_text_to_speech(video_id):
    import os
    import requests
    from django.conf import settings
    from experiments.models import Video
    from constants import TEMP_LOCAL_PATH, FLOW_STATUS, ELEVENLABS_VOICE_ID_MAP

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="Start convert_text_to_speech: {}".format(video_id))

    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        return "Video.DoesNotExist"

    output_file = TEMP_LOCAL_PATH.TEMP_OUTPUT_AUDIO.format(video_id)
    if os.path.isfile(output_file):
        replace_video_audio.delay(video_id=video_id)
        return

    if video.voice_gender == "M":
        voice_id = ELEVENLABS_VOICE_ID_MAP.get(video.voice_gender, {}).get('bipin')
    else:
        voice_id = ELEVENLABS_VOICE_ID_MAP.get(video.voice_gender, {}).get('nissa')
    if not voice_id:
        voice_id = ELEVENLABS_VOICE_ID_MAP.get("default")

    url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}".format(voice_id=voice_id)
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": settings.ELEVEN_LABS
    }
    data = {
        "text": video.translated_text,
        "model_id": "eleven_multilingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    video.status = FLOW_STATUS.TEXT_TO_SPEECH_IN_PROCESS
    video.voice_id = voice_id
    video.save()

    response = requests.post(url, json=data, headers=headers)

    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    video.status = FLOW_STATUS.TEXT_TO_SPEECH_IN_PROCESS
    video.save()

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="End convert_text_to_speech: {}".format(video_id))
    replace_video_audio.delay(video_id=video_id)


@shared_task
def replace_video_audio(video_id):
    import os
    from yt_dlp import YoutubeDL
    from experiments.models import Video
    from constants import TEMP_LOCAL_PATH, FLOW_STATUS

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="Start replace_video_audio: {}".format(video_id))
    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        return
    video.status = FLOW_STATUS.MERGING_AUDIO_VIDEO_IN_PROCESS
    video.save()

    # Step 1: Download the YouTube video
    temp_input_video = TEMP_LOCAL_PATH.TEMP_INPUT_VIDEO.format(video_id)
    if not os.path.isfile(temp_input_video):
        with YoutubeDL({
            'format': 'bestvideo[height<=480][ext=mp4]/best[ext=mp4][height<=480]',
            'outtmpl': temp_input_video,
        }) as ydl:
            ydl.download([video.youtube_url])

    temp_output_video = TEMP_LOCAL_PATH.TEMP_OUTPUT_VIDEO.format(video_id)
    if not os.path.isfile(temp_output_video):
        # Add audio to video
        ffmpeg_command = 'ffmpeg -i "{video_path}" -i "{audio_file}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}"'.format(
            video_path=temp_input_video,
            audio_file=TEMP_LOCAL_PATH.TEMP_OUTPUT_AUDIO.format(video_id),
            output_path=temp_output_video,
        )
        os.system(ffmpeg_command)

    video.status = FLOW_STATUS.MERGING_AUDIO_VIDEO_COMPLETED
    video.save()

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="End replace_video_audio: {}".format(video_id))
    upload_video.delay(video_id=video_id)


@shared_task
def upload_video(video_id):
    import os
    import boto3
    from experiments.models import Video

    from constants import TEMP_LOCAL_PATH, FLOW_STATUS

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="Start replace_video_audio: {}".format(video_id))
    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        return
    video.status = FLOW_STATUS.UPLOADING_VIDEO
    video.save()

    output_file_path = TEMP_LOCAL_PATH.TEMP_OUTPUT_VIDEO.format(video_id)

    s3 = boto3.resource('s3')
    filename = output_file_path.split('/')[1]
    file_object = s3.Object('kuku-hackathon', filename)
    file_object.upload_file(output_file_path, ExtraArgs={'ACL': 'public-read'})
    key = file_object.key

    video.media_key = key
    video.status = FLOW_STATUS.COMPLETED
    video.save()

    send_slack_message.delay(channel="#hackathon-2023-logs", username="Log:{}".format(video_id),
                             text="Video successful completed\n{}".format(video_id, video.get_s3_link()))

    # Removing temporary files
    for path in [TEMP_LOCAL_PATH.TEMP_INPUT_AUDIO, TEMP_LOCAL_PATH.TEMP_OUTPUT_AUDIO,
                 TEMP_LOCAL_PATH.TEMP_INPUT_VIDEO, TEMP_LOCAL_PATH.TEMP_OUTPUT_VIDEO]:
        if not os.path.isfile(path.format(video_id)):
            os.remove(path.format(video_id))

    # import os
    # import googleapiclient.discovery
    # from google.oauth2 import service_account
    #
    # # Set your API credentials file path
    # api_credentials_file = 'path/to/credentials.json'
    #
    # # Set the path of the video file to be uploaded
    # video_file_path = 'path/to/video.mp4'
    #
    # # Set the title, description, and tags for the uploaded video
    # video_title = 'My Uploaded Video'
    # video_description = 'Description of my video'
    # video_tags = ['tag1', 'tag2', 'tag3']
    #
    # # Authenticate and create a YouTube Data API client
    # credentials = service_account.Credentials.from_service_account_file(api_credentials_file, scopes=[
    #     'https://www.googleapis.com/auth/youtube.upload'])
    # youtube = googleapiclient.discovery.build('youtube', 'v3', credentials=credentials)
    #
    # # Create a request body with video details
    # request_body = {
    #     'snippet': {
    #         'title': video_title,
    #         'description': video_description,
    #         'tags': video_tags
    #     },
    #     'status': {
    #         'privacyStatus': 'public'
    #     }
    # }
    #
    # # Create the video insert request
    # insert_request = youtube.videos().insert(
    #     part='snippet,status',
    #     body=request_body,
    #     media_body=googleapiclient.http.MediaFileUpload(video_file_path, chunksize=-1, resumable=True)
    # )
    #
    # # Execute the video insert request
    # response = None
    # while response is None:
    #     status, response = insert_request.next_chunk()
    #     if status:
    #         print(f"Uploaded {int(status.progress() * 100)}%")
    #
    # # Print the uploaded video details
    # print("Video uploaded! Video ID: {response['id']}")


@shared_task
def send_slack_message(channel, username, text):
    import requests
    from django.conf import settings
    response = requests.post('https://slack.com/api/chat.postMessage', json={
        "channel": channel,
        "username": username,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text
                }
            }
        ]
    }, headers={
        'Authorization': 'Bearer %s' % settings.SLACK_BOT_API_TOKEN,
        'Content-Type': 'application/json;charset=UTF-8'
    })
    json_data = response.json()
    if not json_data['ok']:
        return False
    return True
