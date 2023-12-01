import subprocess
import logging
import json
from services import convert_audio
from config_data.config import OPENAI_API_KEY
from lexicon.lexicon import LEXICON
from random import choice

TRANSCRIPTIONS_URL = 'https://api.openai.com/v1/audio/transcriptions'
CHAT_URL = 'https://api.openai.com/v1/chat/completions'

logging.basicConfig(level=logging.WARNING)
logger_ai_service = logging.getLogger(__name__)


def transcribe_audio_to_text(file_bytes: bytes = None,
                             file_name: str = 'default_name.mp3') -> str:
    logger_ai_service.warning('Send mp3 to OpenAI')

    result_text = ''
    path_save_txt = f"temp/{file_name.split('.')[0]}.txt"
    model = "whisper-1"

    dict_res = curl_post_request_sound_transcribe(file_bytes, model=model)

    if "text" in dict_res:
        result_text = dict_res["text"] if dict_res["text"] else choice(LEXICON["no_words"])

    elif "error" in dict_res and "Invalid file format" in dict_res["error"]["message"]:
        try:
            file_bytes = convert_audio.convert_audio_to_mp3(file_bytes=file_bytes, speed=1)
            dict_res = curl_post_request_sound_transcribe(file_bytes, model=model)
            result_text = dict_res.get("text", choice(LEXICON["another_wrong"]))

        except BaseException as e:
            logger_ai_service.warning("Something wrong \n", "Exception: ", e)

    elif "error" in dict_res and "server_error" in dict_res["error"]["message"]:
        logger_ai_service.warning("Server OpenAi Error")
        result_text = choice(LEXICON["server_error"])

    with open(path_save_txt, "w", encoding="utf-8") as f:
        f.write(result_text)

    logger_ai_service.warning('Successfully convert to text')

    return path_save_txt


def text_request_to_open_ai(text: str = "Say me something good!") -> str:
    model = "gpt-3.5-turbo"
    dict_answer = curl_post_text_request(text=text, model=model)
    text_answer = dict_answer.get("choices", [{}])[0].get("message", {}).get("content", '')

    if "error" in dict_answer:
        logger_ai_service.warning(dict_answer["error"]["message"])
        text_answer = choice(LEXICON["another_wrong"])

    return text_answer


def curl_post_request_sound_transcribe(sound_bytes: bytes = None,
                                       model="whisper-1") -> dict:
    curl_command = ['curl', '--request', 'POST',
                    '--url', f'{TRANSCRIPTIONS_URL}',
                    '--header', f'Authorization: Bearer {OPENAI_API_KEY}',
                    '--header', 'Content-Type: multipart/form-data',
                    '--form', f'file=@-;filename=file_name.mp3',
                    '--form', f'model={model}']

    res = subprocess.run(curl_command,
                         input=sound_bytes,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, check=True)

    if res.returncode == 0:
        return json.loads(res.stdout)
    else:
        logger_ai_service.warning(f"Error return code \n{res.returncode}")
        return {"error": {"message": "server_error"}}


def curl_post_text_request(text, model="gpt-3.5-turbo", temperature: float = 1) -> dict:
    data = json.dumps({
        "model": f"{model}",
        "messages": [
            {
                "role": "user",
                "content": text,
            }
        ],
        "temperature": temperature
    })

    curl_command = ['curl', '--request', 'POST',
                    '--url', f'{CHAT_URL}',
                    '-H', "Content-Type: application/json",
                    '-H', f'Authorization: Bearer {OPENAI_API_KEY}',
                    '-d', f'{data}']

    result = subprocess.run(curl_command, stdout=subprocess.PIPE, check=True)

    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        return {"error": {"message": f"System code mistake \n{result.returncode}"}}
