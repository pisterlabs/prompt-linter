import openai
import os
from youtube_transcript_api import YouTubeTranscriptApi
import re


#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key =  "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"

def get_video_id_from_url(url):
    video_id = None
    pattern = re.compile(r"(?:https?:\/\/)?(?:www\.)?youtu(?:\.be\/|be\.com\/watch\?v=)([\w\-]+)")
    match = pattern.match(url)

    if match:
        video_id = match.group(1)

    return video_id


def get_video_transcript(url):
    video_id = get_video_id_from_url(url)

    if not video_id:
        print("Invalid YouTube video URL.")
        return

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        #print(transcript_text)
        return transcript_text
    except Exception as e:
        #print(f"Error: {str(e)}")
        return f"Error: {str(e)}"

def educationChecker(title):
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You have to tell whether the video is educational or not basis on the title. Just write educational or not educational.Gym exercises and trailers are not educational"},
    {"role":"user", "content": title}
  ]
  )
  return completion.choices[0].message["content"]

def learningOutcomes(transcript):
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "I am giving you a transcript of a Youtube video. Your task is to tell me how by watching this video, my understanding of the world has changed. Tell it in 3 bullet points. Use jargons. Write each point, as if you are explaining how my understanding has changed. Use max 10 words for each point. Write in past tense, i.e. understood rather than understanding. use max 10 words per bullet point. Use '-' for bullet points, don't use numbered list.Tell exactly the learning outcomes of the video,keep it very very short. Tell it like understood the impacts of etc.Always complete the sentences. Always give 3 points.If subtitles not available just say not available"},
    {
      "role": "user", "content": transcript
    }
  ]
  )
  return completion.choices[0].message["content"]


def finalReport(learnings):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"""I am giving you a list of stuff I have learnt from youtube.
You have to distribute it in History, Geography, Civics, General Knowledge, Economics, Tech, Political Science , Skills, Consumer Psychology, Health and then tell me in bullet points how my knowledge has grown in those subjects. Write the first line congratulating me to have learnt all the topics also mention the topics very briefly there.
My name is faraaz\n{learnings}""",
            },
            {
                "role": "user",
                "content": "Tell me what all I have learnt. Use a lot of jargons, divide it in the subjects mentioned and then add there. Only mention the subjects where knowledge has been gained. Do not mention the subjects where specific knowledge has not been gained. First mention the subject, then tell it in bullet points how my understanding has improved.Remember the bullet points.",
            },
        ],
    )
    return completion.choices[0].message["content"]

def YTgenerator(link):
  transcript = get_video_transcript(link)
  outcomes = learningOutcomes(transcript)
  return outcomes

def YTlearner(link):
  transcript = get_video_transcript(link)
  print("The transcript is this "+ transcript)
  outcomes = stepsToLearn(transcript)
  return outcomes

def stepsToLearn(transcript):
  completion = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "I am giving you a transcript of a Youtube tutorial video. You have to list out the steps followed in the tutorial in great detail. Guide me on how can I complete the project shown in video"},
    {
      "role": "user", "content": transcript
    }
  ]
  )
  return completion.choices[0].message["content"]