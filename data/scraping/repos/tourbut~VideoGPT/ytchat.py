from pytube import YouTube

def download_youtube_video(url, filename='temp_video.mp4'):
    try:
        # Create a YouTube object with the URL
        yt = YouTube(url)
        # Select the first stream: usually the best available
        video_stream = yt.streams.filter(file_extension='mp4').first()
        if not video_stream:
            print("No mp4 video stream available")
            return False

        # Set the filename
        video_stream.download(filename=filename)
        print("Download complete!")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Example usage:
url = ''
#download_youtube_video(url, filename='temp_video.mp4')

from moviepy.editor import *

def convert_mp4_to_mp3(mp4_file_path, mp3_file_path):
    try:
        # 비디오 클립 로드
        video_clip = VideoFileClip(mp4_file_path)
        # 오디오 추출 및 MP3 파일로 저장
        video_clip.audio.write_audiofile(mp3_file_path)
        print("MP3 변환 완료!")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        
#convert_mp4_to_mp3('temp_video.mp4', 'temp_audio.mp3')


import openai

# OpenAI API 키 설정
openai.api_key = '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'

def transcribe_audio_to_text(audio_file_path, text_file_path):
    try:
        # 오디오 파일 열기
        with open(audio_file_path, 'rb') as audio_file:
            # Whisper API를 사용하여 오디오 파일 전사
            transcript_response = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
            )

            # 전사된 텍스트 가져오기
            transcribed_text = transcript_response

            # 텍스트 파일로 저장
            with open(text_file_path, 'w') as text_file:
                text_file.write(transcribed_text)
            
            print(f"전사된 텍스트가 {text_file_path}에 저장되었습니다.")
            return transcribed_text
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return None

# Example usage:
#transcribed_text = transcribe_audio_to_text('temp_audio.mp3','temp_text.txt')
#print(transcribed_text)


def get_context_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            context = file.read()
        return context
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None
    except Exception as e:
        print(f"파일을 읽는 동안 오류가 발생했습니다: {e}")
        return None
    
def chat_with_gpt4(context, user_message):
    try:
        # API 호출을 위한 메시지 리스트 생성
        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant."},
        ]

        # 컨텍스트를 시스템 메시지로 추가 (선택적)
        if context:
            messages.append({"role": "system", "content": context})

        # 사용자 메시지 추가
        messages.append({"role": "user", "content": user_message})
        
        # GPT-4를 사용하여 대화 완성 API 호출
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages
        )
        #print(response)
        # 응답 추출
        answer = response.choices[0].message
        return answer
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return None
    
context = get_context_from_file('temp_text.txt')

answer = chat_with_gpt4(context, "무슨 내용이야?")
print(answer)