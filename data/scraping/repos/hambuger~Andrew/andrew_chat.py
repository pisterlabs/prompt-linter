import time
import os
import traceback
from config.global_logger import logger
from voice_util.asr.asr_invoke import audio_to_text
from openai_util.s_auto_gpt import run_conversation_v2, push_message
from voice_util.kws.audio_kws import get_audio, set_user_input_str
from voice_util.tts.voice_tts import text_2_audio, stop_speak
from database_util.redis.redis_client import api_key_manager
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(10)
input_str = None

# Records the last time a valid input was detected
last_input_time = 0
# Whether voice chat is activated
audio_active = False
# Default audio file path
file_path = 'tmp/audio.wav'
# Record the ID of the last dialog response message
parent_id = '0'


def get_input():
    global input_str
    while True:
        input_str = input()
        set_user_input_str(input_str)
        if not input_str:
            continue
        if input_str.lower() == 'stop':
            stop_speak()
            input_str = None
            continue


executor.submit(get_input)
logger.info("Andrew chat started")
print("Andrew chat started")
while True:
    try:
        audio_text = None
        if get_audio(audio_active, file_path, last_input_time):
            # Execute ASR and print the result
            audio_text = audio_to_text(file_path)
            logger.debug(f"Recognize speech：{audio_text}")
            last_input_time = time.time()
            audio_active = True
        if not audio_text and not input_str:
            continue
        bye_word = os.getenv('BYE_WORD', 'goodbye')
        if audio_text and bye_word in audio_text.lower():
            logger.debug("goodbye!")
            audio_active = False
            parent_id = '0'
            text_2_audio(bye_word)
            print('\033[32m' + "goodbye!" + '\033[0m')
            if api_key_manager.get_key_value('AUDIO_KEY') == os.getenv('OS_NAME'):
                api_key_manager.delete_key('AUDIO_KEY')
            continue
        audio_text = (f"""{input_str}\n{audio_text}""" if audio_text else input_str) if input_str else audio_text
        if audio_text != input_str:
            print('\033[32m' + f"{os.getenv('MY_NAME')}：{audio_text}" + '\033[0m')
        push_message({"role": "user", "content": audio_text})
        (answer, msg_Id) = run_conversation_v2(audio_text, parent_id)
        if msg_Id:
            parent_id = msg_Id
        if not answer:
            continue
        push_message({"role": "assistant", "content": answer})
        logger.debug(f"AI response：{answer}")
        print('\033[31m' + f"AI response：{answer}" + '\033[0m')
        text_2_audio(answer)
        last_input_time = time.time()
    except Exception as e:
        logger.error(e)
        # print exception stack
        logger.error(traceback.format_exc())
        continue
    finally:
        if api_key_manager.get_key_value('AUDIO_KEY') == os.getenv('OS_NAME'):
            api_key_manager.delete_key('AUDIO_KEY')
