import threading
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
from io import BytesIO
import openai
import random
import time

SAMPLING_FREQUENCY = 16000
DURATION_PER_CHUNK = 2
INPUT_INDEX = 1
CHATGPT_API_KEY = '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'

MARA_BRAIN = "gpt-3.5-turbo"
MARA_RUDE_ON = True
MARA_SPEED = 1.2
MARA_SAMNIENG = 'th'

def play_wtf(text):

    ml_thread = threading.Thread(target=gpt_kidspap, args=([text]))
    ml_thread.start()

    prefix = [
        "เดี๋ยวๆๆ คุณหยุดพูดก่อน ",
        "ฉันว่ามันไม่ได้ คุณไปพักก่อนนะ ",
        "คุณจะพรีเซ้นแบบนี้จริงๆหรอฉันถามจริง ",
        "พรีเซ้นอะไรของคุณ ฉันหล่ะปวดหัวจริงๆ "
    ][random.randint(0, 3)]

    suffix = [
        "ขอฉันคิดแป๊ปนะ รอฉันก่อน",
        "รอฉันเดี๋ยวครับ รอฉันก่อน",
        "ขอฉันพิจารณาก่อนครับ รอฉันก่อน"
    ][random.randint(0, 2)]

    text2speech(prefix + suffix)
    ml_thread.join()

def gpt_kidspap(text):
    res = chatgpt_generate(text=text)
    text2speech(res)

def chatgpt_generate(text):

    t1 = time.time()

    openai.api_key = CHATGPT_API_KEY

    messages = [
        {"role": "system", "content": "คุณคือคนที่คอยติเนื้อหาการนำเสนองานของฉันอย่างตรงไปตรงมาอยู่บ่อยครั้ง แต่คุณก็ยังให้คำแนะนำ และพูดแบบไม่รักษาน้ำใจ พูดห้วนๆ"},
        {"role": "user", "content": f"คิดคำติจากคำเหล่านี้ '{text}' และให้ทางออกว่าฉันควรพูดอะไรแทน โดยพูดให้สั้นที่สุด"}
    ]

    completion = openai.ChatCompletion.create(
        model=MARA_BRAIN,
        messages=messages
    )

    return completion.choices[0].message.content

def text2speech(text):

    if MARA_RUDE_ON:
        text = text.replace('ฉัน', 'กู').replace('คุณ', 'มึง').replace('ครับ', 'นะไอ้สัตว์').replace('ค่ะ', 'นะไอ้สัตว์').replace('ขอบมึง', 'ขอบคุณ')

    tts = gTTS(text, lang=MARA_SAMNIENG)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    sound = AudioSegment.from_file(fp, format="mp3")
    sound = sound.speedup(MARA_SPEED, 150, 25)

    play(sound)


if __name__ == '__main__':
    text2speech('เขย่ากันด้วยนะ ไอสัส')