from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pytube import YouTube
from moviepy.editor import *
from youtube_transcript_api import YouTubeTranscriptApi
import openai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin (Not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

title = ""
api_key = "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"


def download_youtube_video(url, download_path):
    yt = YouTube(url)
    yt_stream = yt.streams.filter(file_extension="mp4").first()
    yt_stream.download(output_path=download_path, filename="video.mp4")
    title = yt.title


# Function to extract audio and save it as a wav file
def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)


# Function to download captions using youtube-transcript-api
def download_captions(video_id, lang_code="en"):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=[lang_code]
        )
        srt_captions = ""
        for entry in transcript:
            start = entry["start"]
            duration = entry["duration"]
            end = start + duration
            text = entry["text"].replace("\n", " ")

            srt_captions += f"{start} --> {end}\n{text}\n\n"
        return srt_captions
    except Exception as e:
        return f"An error occurred: {e}"


def filter_captions(captions):
    lines = captions.strip().split("\n")
    text_lines = [line for i, line in enumerate(lines) if (i - 1) % 3 == 0]
    return " ".join(text_lines)


def fetch_gpt3_response(prompt, max_tokens=1000):
    openai.api_key = api_key

    # Sending the prompt to the GPT-3.5-turbo engine
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Extracting the generated text and storing it in a string
    generated_text = response.choices[0].text.strip()

    return generated_text


@app.get("/getVideoSummary")
async def get_video_summary(videoUrl: str = Query(...), videoLang: str = Query("en")):
    print(f"GET request received. videoUrl={videoUrl},videoLang={videoLang}")
    try:
        # Your existing code, but now use `videoUrl` and `videoLang` instead of args
        video_id = videoUrl.split("v=")[1].split("&")[0]
        temp_folder = "temp_folder/"
        video_path = temp_folder + "video.mp4"
        audio_path = temp_folder + "audio.wav"

        print("Downloading Captions...")
        captions = download_captions(video_id, videoLang)
        print("Filtering Captions...")
        filtered_captions = filter_captions(captions)

        prompt = (
            f"Please summarize the following video script in a few sentences. The name of the YouTube Video is {title}. Filter out only the information and do not include any possible sponsored segments. ABSOLUTELY NEVER use phrases like 'the video explains', or 'the script argues', instead just focus on the raw information, as if you were asked to explain the topic to someone without sounding like you learnt it in a YouTube video. Lastly again, completely ignore anything that seems like a sponsorship or a sponsored segment, or anything where someone directly talks about a product. Don't leave out any facts in the video. Here is the related video script:\n\n"
            + filtered_captions
            + "\n\nSummary:"
        )

        print("Fetching OpenAI Response...")
        response = fetch_gpt3_response(prompt)

        print("Returning GET request...")
        return JSONResponse(content={"summary": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
