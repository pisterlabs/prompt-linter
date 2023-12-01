import openai
import sounddevice as sd
from scipy.io.wavfile import write
import os
from dotenv import load_dotenv

# 録音の設定
SAMPLE_RATE = 16000     # サンプリングレート
DURATION = 3            # 録音時間（秒）
CHANNELS = 1            # モノラル

# 録音ファイルのパス
REC_FILE = 'sound/recorded.wav'

# OpenAIのAPIキーを設定
load_dotenv('.env') 
openai.api_key = os.environ.get("OPEN_AI_API_KEY")

# 録音実行
print('Recording...')
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
sd.wait()
print('Recording stop')

# wavファイルに保存
write(REC_FILE, SAMPLE_RATE, recording)

audio_file= open(REC_FILE, "rb")

# Whisper APIでテキストに変換
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript.text)