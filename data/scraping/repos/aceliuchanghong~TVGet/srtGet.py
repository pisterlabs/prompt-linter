from crawl.spiderDealer.checkPath import check
import os
import sys
from openai import OpenAI
import io
import httpx

from crawl.spiderDealer.srt2Txt import modify_subtitle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def mp32srt(result, name=None):
    # Get file name
    mp3path = result.mp3path
    output_file = mp3path.split('/')[-1].replace("mp3", "srt")

    if name is not None:
        output_file = name

    proxyHost = "127.0.0.1"
    proxyPort = 10809

    client = OpenAI(http_client=httpx.Client(proxies=f"http://{proxyHost}:{proxyPort}"))
    client.api_key = os.getenv("OPENAI_API_KEY")

    relative_path = '../../crawl/files/srt/'
    check(relative_path)
    realFilePath = relative_path + output_file
    if not os.path.exists(realFilePath):
        prompt = "请返回中文字幕,这是一段关于中国外交部的发言稿,主要包括" + result.title
        # print(realFilePath, prompt)

        try:
            file = open(mp3path, "rb")
            transcript = client.audio.transcriptions.create(model="whisper-1", language='zh', file=file,
                                                            response_format="srt",
                                                            prompt=prompt)
            # transcript = openai.Audio.transcribe("whisper-1", file, response_format="srt",
            #                                prompt=prompt)
            with open(realFilePath, 'w', encoding='utf-8') as f:
                f.write(transcript)
            print("srt from gpt SUC")

        except Exception as e:
            print("Srt deal Error:", e)
            # remove file
            if os.path.exists(realFilePath):
                os.remove(realFilePath)
            print("\n")
            print(result)
            return None
    try:
        with open(realFilePath, 'r', encoding='utf-8') as file:
            srt_contents = file.read()
            char_count = len(srt_contents)
            if char_count < 10:
                print("ERR: srt is none")
                return None
    except Exception as e:
        print("Srt deal Error 2:", e)
        return None
    last_path = modify_subtitle(realFilePath, 13)
    print("srt modify SUC")

    return last_path
