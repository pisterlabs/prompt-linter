from youtube_transcript_api import YouTubeTranscriptApi
import openai

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        text = " ".join([line["text"] for line in transcript])  
    except:
        text = '해당 영상은 자막을 제공하지 않습니다.'
    return text

def get_video_summary(api_key, script):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "콘텐츠를 한국어 3줄로 요약해 줘, 각 줄은 숫자로 시작하고 문단의 끝에는 줄바꿈 표시를 넣어 줘"},
            {"role": "user", "content": script}
        ],
        temperature=0.2,
        max_tokens=256
    )
    return response['choices'][0]['message']['content']  # 응답에서 원본 소스 코드만 반환

def get_summary(api_key, video_id):
    script = get_video_transcript(video_id)
    summary = get_video_summary(api_key, script)
    return summary