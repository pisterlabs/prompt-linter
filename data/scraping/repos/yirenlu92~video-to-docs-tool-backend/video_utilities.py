import tempfile
import requests
import cv2
import stable_whisper
import openai
import os


def chunk_video_and_merge_transcript(video_path):
     # chunk up video file
    chunk_up_video_file(video_path)

    # get the paths of all the video chunks
    video_chunk_paths = []
    for filename in os.listdir("video_chunks"):
        video_chunk_paths.append("video_chunks/" + filename)
    
    whole_transcript = ""
    for path in video_chunk_paths:
        # transcribe the video chunk
        transcript_chunk = transcribe_video_whisper_api(path)
        whole_transcript += transcript_chunk

    print("open ai transcript:")
    print(whole_transcript)
    return whole_transcript


# def chunk_up_video_file(video_path):

#     song = AudioSegment.from_file(video_path)

#     # PyDub handles time in milliseconds
#     ten_minutes = 10 * 60 * 1000

#     # chunk up the song in increments of ten minutes
#     for i, chunk in enumerate(song[::ten_minutes]):
#         with open(f"video_chunks/video_{i}.mp4", "wb") as f:
#             chunk.export(f, format="mp4")

def transcribe_video_whisper(video_url):

    print("beginnign of transcription")
    model = stable_whisper.load_model('base')

    # modified model should run just like the regular model but accepts additional parameters
    result = model.transcribe(video_url)

    sentences = result.to_srt_vtt(word_level=False)

    print("got sentences back in the transcription")

    print(sentences)
    return sentences

def transcribe_video_whisper_api(video_path):

    openai.api_key = os.getenv("OPENAI_API_KEY")
    audio_file = open(video_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']


def download_video(url):
    response = requests.get(url)
    video_data = response.content
    return video_data

def extract_screenshot_images(video_data, timestamps):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_data)
        temp_file.flush()

        video = cv2.VideoCapture(temp_file.name)

        screenshots = []
        for timestamp in timestamps:
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = video.read()

            if ret:
                img_encoded = cv2.imencode(".png", frame)[1].tobytes()
                screenshots.append(img_encoded)

        video.release()

        os.unlink(temp_file.name)

    return screenshots

def upload_screenshots_to_gcs(screenshots, folder_name, bucket):
    public_urls = []
    for idx, screenshot in enumerate(screenshots):
        file_name = f"{folder_name}/screenshot_{idx}.png"
        blob = bucket.blob(file_name)
        blob.upload_from_string(screenshot, content_type="image/png")
        blob.make_public()
        public_urls.append(blob.public_url)

    return public_urls

def transcript_to_blog_post_with_chatgpt(transcript):

    prompt = f"""
    This is a transcript of a video demo of how to monitor Stripe data with Bigeye. Please lightly rewrite the transcript into a tutorial with an introduction, and appropriate instructions formatted in markdown paragraphs separated by newlines. 
    """

    messages=[
        {"role": "system", "content": "You are transcribing the audio of a software demo video. The timestamps correspond to the times that the phrases were spoken."},
        {"role": "user", "content": prompt},
    ]

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.1,
    messages=messages,
    timeout=30
    )

    return completion.choices[0].message["content"]

def transcript_to_tutorial_instructions_with_chatgpt(transcript):

    prompt = f"""
    Please lightly rewrite the transcript into a step by step tutorial for the software, mapping each of the steps to a timestamp in the video that it would correspond to. Please only include the instructions in the transcript. Don't make up additional instructions. You may want to combine a few of the steps for brevity. The output should show one text/timestamp pair per line. Here's an example:\n\n    Give instruction one -- 00:00:00,640 --> 00:00:03,320\n\nGive instruction 2 -- 00:00:03,425 --> 00:00:05,425\n\n...\n\n. Now please edit this transcript:\"\"\"{transcript}\"\"\"
    """

    messages=[
        {"role": "system", "content": "You are transcribing the audio of a software demo video. The timestamps correspond to the times that the phrases were spoken."},
        {"role": "user", "content": prompt},
    ]

    completion = openai.ChatCompletion.create(
    model="gpt-4-0613",
    temperature=0.1,
    messages=messages,
    timeout=30
    )

    return completion.choices[0].message["content"]