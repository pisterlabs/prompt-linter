import requests
from user_files.config import *
from utils.katakana import *
import re
import requests
import os
from openai import OpenAI
from pydub import AudioSegment

def tts_get_en(tts, speaker, pitch):
    import torch
    lang = "en"
    model = "v3_en"
    if not speaker:
        speaker = "en_18"
    text_fill = remove_act(tts)
    if not text_fill:
        if not tts:
            tts = "Error error"
        text_fill = tts

    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'user_files/model.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file(f'https://models.silero.ai/models/tts/{lang}/{model}.pt',
                                    local_file)  

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    sample_rate = 48000
    model.save_wav(text=tts,
                    speaker=speaker,
                    sample_rate=sample_rate, audio_path='user_files/test.wav')
    
    input_wav_file = 'user_files/test.wav'
    output_ogg_file = 'user_files/ai_voice_msg.ogg'
    audio = AudioSegment.from_wav(input_wav_file)
    pitching = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * (2 ** (pitch / 12.0)))
    })
    pitching.export(output_ogg_file, format="ogg")

def tts_get(text, speaker, pitch, intonation_scale, speed, console_log):
    text_fill = remove_act(text)
    if not text_fill:
        if not text:
            text = "エラー エラー"
        text_fill = text
    cnv_text = romaji_to_katakana(text_fill)
    url = f"https://deprecatedapis.tts.quest/v2/voicevox/audio/?key={vv_key}&text={cnv_text}&speaker={speaker}&pitch={pitch}&intonationScale={intonation_scale}&speed={speed}"
    
    response = requests.get(url)
    
    if response.status_code == 200:

        with open('user_files/ai_voice_msg.ogg', 'wb') as f:
            f.write(response.content)
        if console_log:
            print(f"Voice đã được tải về thành công.")
    else:
        print(f"Lỗi khi tạo voice, mã lỗi: {response.status_code}")

def oa_tts(text, speaker, pitch):
    text = remove_act(text)
    if not text:
        text = "うーん？"
    client = OpenAI(api_key=openai_key_2, timeout=60)
    path = "user_files/ai_voice_msg.wav"
    response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input=text
    )

    ipath = 'user_files/ai_voice_msg.wav'
    opath = 'user_files/ai_voice_msg.ogg'
    response.stream_to_file(path)
    sound = AudioSegment.from_file(ipath)
    pitching = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * (2 ** (pitch / 12.0)))
    })
    pitching.export(opath, format="ogg")

def remove_act(text):
    text = re.sub(r'\*([^*]+)\*', '', text)
    text = re.sub(r'\([^)]+\)', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return text

if __name__ == "__main__":
    tts_get()