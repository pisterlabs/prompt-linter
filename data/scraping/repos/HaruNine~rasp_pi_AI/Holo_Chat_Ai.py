//this is all of main code 

import os
import io
import speech_recognition as sr
from google.cloud import texttospeech
from google.cloud import speech_v1p1beta1 as speech
import openai
import wave
import pyaudio
import pygame
import numpy as np

# Pygame 초기화
pygame.init()

# 화면 설정
width, height = 864, 480  # 임시 크기 (전체 화면으로 설정된 이후에는 무시됨)
screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
pygame.display.set_caption('Audio Visualization')

# 마우스 숨기기
pygame.mouse.set_visible(False)

# 파이게임 시계 설정
clock = pygame.time.Clock()  # 시계 생성

# Google Cloud 서비스 계정 키 파일 경로 설정
keyfile_path = "GOOGLE CLOUD API.json 위치경로"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = keyfile_path

# OpenAI API 키 설정
with open('openai_session 위치경로', 'r') as file:
    api_key = file.read().strip()
openai.api_key = api_key

# Google Cloud Text-to-Speech 클라이언트 초기화
client = texttospeech.TextToSpeechClient()

# 키워드
KEYWORD = "라니"

# 음성 인식 객체 생성
recognizer = sr.Recognizer()


def record_and_recognize():
    with sr.Microphone() as source:
        print("말씀해주세요...")

        # 주변 소음을 측정하여 조정
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while True:
            try:
                audio = recognizer.listen(source)

                # 음성을 텍스트로 변환
                text = recognizer.recognize_google(audio, language="ko-KR")
                print(f"인식된 텍스트: {text}")

                # 특정 키워드 감지
                if KEYWORD in text:
                    print(f"키워드 '{KEYWORD}' 감지됨!")
                    execute_voice_chat()
                    break  # 키워드를 찾았으므로 반복 중단
                else:
                    print("키워드가 감지되지 않았습니다.")

            except sr.UnknownValueError:
                record_and_recognize()
            except sr.RequestError as e:
                print(f"Google 음성 API 요청 오류: {e}")


def execute_voice_chat():
    try:
        speak_response("네 라니입니다")  # "네" 음성 출력
        main()
    except Exception as e:
        print(f"음성 대화 중 오류 발생: {e}")


def speak_response(response_text):
    # Google Cloud Text-to-Speech API를 사용하여 음성 생성
    synthesis_input = texttospeech.SynthesisInput(text=response_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Wavenet-A",
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # 음성을 WAV 파일로 저장
    with open("response.wav", "wb") as out:
        out.write(response.audio_content)

    # WAV 파일을 스피커로 출력
    play_game("response.wav")


# 파이오디오 녹음 함수
def record_audio(file_path, record_seconds=3):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# Google Speech-to-Text API 사용해서 text변환
def transcribe_audio(file_path, language_code="ko-KR"):
    client = speech.SpeechClient()

    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript


# OpenAI API ChatGPT 설정 및 구동
def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()


# Google Text-to-Speech API 사용해서 오디오로 변환
def text_to_speech(text, output_file="output.wav"):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Wavenet-A",
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)


# 키워드 호출로 돌아가게하기
def start_voice_chat():
    try:
        record_and_recognize()
    except Exception as e:
        print(f"Error starting voice chat: {e}")


def play_game(file_path):
    # 오디오 파일 로드
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()

    # 오디오 파일의 총 프레임 수 확인
    total_frames = wf.getnframes()

    # buffer_size 설정 (오디오 파일의 총 프레임 수 이상으로 설정하지 않도록 주의)
    buffer_size = min(total_frames, 2048)
    samples = np.zeros(buffer_size)
    x = np.arange(0, buffer_size)
    line = pygame.Surface((width, height))

    # 사운드 스트림 설정
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 무지개 색상 배열 정의
    rainbow_colors = [
        (148, 0, 211),  # 보라
        (75, 0, 130),  # 남색
        (0, 0, 255),  # 파랑
        (0, 255, 0),  # 초록
        (255, 255, 0),  # 노랑
        (255, 165, 0),  # 주황
        (255, 0, 0)  # 빨강
    ]

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 오디오 읽기
        data = wf.readframes(buffer_size)
        if not data:
            break

        samples = np.frombuffer(data, dtype=np.int16)

        # 시각화 업데이트
        line.fill((0, 0, 0))  # 검은 배경
        for i in range(min(buffer_size, width)):  # 최소값 사용
            if i < len(samples):  # 길이를 확인하여 오류 방지
                y = height // 2 - samples[i] / 90

                # 무지개 색상 선택
                rainbow_index = int(i / buffer_size * len(rainbow_colors))
                color = rainbow_colors[rainbow_index]

                pygame.draw.aaline(line, color, (i, height // 2), (i, y), 1)  # 무지개 선 (안티 에일리어싱 적용)

        # 화면에 표시
        screen.blit(line, (0, 0))
        pygame.display.flip()

        # 사운드 출력
        stream.write(data)

        clock.tick(60)  # 초당 60프레임으로 제한

    # 종료 시 정리
    stream.stop_stream()
    stream.close()
    p.terminate()


# Main function
def main():
    while True:
        # Record audio
        audio_file_path = "input.wav"
        record_audio(audio_file_path)

        # Transcribe audio to text
        user_input = transcribe_audio(audio_file_path)

        if not user_input:
            start_voice_chat()
            break

        print("You:", user_input)

        # Chat with GPT
        gpt_response = chat_with_gpt(f"You: {user_input}\nBot:")
        print("Bot:", gpt_response)

        # Convert GPT response to speech
        text_to_speech(gpt_response)

        # Play the response
        play_game("output.wav")


if __name__ == "__main__":
    record_and_recognize()
