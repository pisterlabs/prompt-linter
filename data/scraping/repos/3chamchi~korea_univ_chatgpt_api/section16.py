# Section 16
# 오디오 변환 프로젝트

# 1. API 연동 준비
import openai

openai.api_key = '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'

# 2. 오디오 파일 읽기
# 파일 경로
#  - Windows : open(r'media\짧은 대화의 응답_22006-0337_통파일.mp3', 'rb')
#  - Mac : open('media/짧은 대화의 응답_22006-0337_통파일.mp3', 'rb')
audio_file = open('media/짧은 대화의 응답_22006-0337_통파일.mp3', 'rb')

# 3. API 요청
response = openai.Audio.transcribe('whisper-1', audio_file)  # 음성 파일의 대화를 해당 언어의 텍스트로 변환
# response = openai.Audio.translate('whisper-1', audio_file)  # 음성 파일의 대화를 영어 텍스트로 변환

# 4. API 응답 확인
print(response)
print(response['text'])
