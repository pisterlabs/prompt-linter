from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import openai


def get_transcript(link):
    # Retrieves the video transcript
    video_id = link.split("=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    print("here")
    result = ""
    for i in transcript:
        result += " " + i["text"]
    return result


def divide_transcript(transcript, chunk_size=0):
    # Determines the chunk_size
    length = len(transcript)
    if length <= 5000:
        chunk_size = 1000
    if length > 5000 and length <= 15000:
        chunk_size = 5000
    if length > 15000 and length <= 30000:
        chunk_size = 8000
    if length > 30000:
        chunk_size = 10000

    # Divide the transcript into smaller chunks
    chunks = []
    num_chunks = len(transcript) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(transcript[start:end])

    return chunks


def summarize_transcript(transcript):
    # Divide the transcript into smaller chunks
    chunks = divide_transcript(transcript)

    # Summarizing the chunks
    summaries = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Summarize this piece "
                 + "of a video transcript: " + chunk}
            ],
            temperature=1,)
        summaries.append(response["choices"][0]["message"]["content"])

    combined_summary = ' '.join(summaries)

    # Summarizing the combined_summary
    response2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Connect this collection of summaries "
             + "into one fluid summary" + combined_summary}
        ],
        temperature=1,)
    final_sum = response2["choices"][0]["message"]["content"]
    return final_sum


def get_video_title(video_url):
    try:
        yt = YouTube(video_url)
        video_title = yt.title
        return video_title
    except Exception as e:
        print(f"Error: {e}")
        # return None
