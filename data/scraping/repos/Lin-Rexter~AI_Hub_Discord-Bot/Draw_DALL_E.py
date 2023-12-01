import os
import openai
from pathlib import Path
from dotenv import load_dotenv

# take environment variables from .env
env_path = os.path.join(Path(__file__).resolve().parents[2], '.env')

load_dotenv(env_path)

# take OpenAI API Key from environment variables
Openai_API_Key = os.getenv('OPENAI_API_KEY') or None


def handle_error(e) -> list:
    ERROR_MESSAGES = {
        openai.error.APIError: "Oops❗ OpenAI的伺服器出了問題，請短暫等待後重試!",
        openai.error.TryAgain: "Oops❗ 請再重新試一次!",
        openai.error.Timeout: "Oops❗ 請求超時，有可能是網路問題，請再重新試一次!",
        openai.error.APIConnectionError: "Oops❗ 無法與OpenAI連線，有可能是網路問題，請再重新試一次!",
        openai.error.InvalidRequestError: "Oops❗ 發送的數據不完整或缺少了參數，請查看API Key是否正確!",
        openai.error.AuthenticationError: "Oops❗ 你的API Key似乎無效或過期，你可能需要再重新生成一個或是重新輸入正確的API Key!",
        openai.error.RateLimitError: "Oops❗ 慢一點，發送的速率過快!",
        openai.error.ServiceUnavailableError: "Oops❗ OpenAI的伺服器無法處理請求，目前可能流量過高，請稍後再重試!",
        UnicodeEncodeError: "Oops❗ 請確認輸入的API Key格式是否正確!"
    }

    print("\nError:", e)
    if type(e).__name__ in ERROR_MESSAGES:
        return ["Error", ERROR_MESSAGES[type(e)]]
    else:
        return ["Dangerous", "Oops❗ 發生了例外錯誤..."]


def DALL_E_Reply(**kwargs) -> list:
    try:
        for key, value in kwargs.items():
            globals()[key] = value

        openai.api_key = api_key or Openai_API_Key or None

        if openai.api_key == None:
            return ["Error", "Oops❗ 尚未設定API Key"]
        else:
            size_type = {
                256: "256x256",
                512: "512x512",
                1024: "1024x1024"
            }

            images_list = []
            response = openai.Image.create(
                prompt = prompts,       # A text description of the desired images
                n = parameter,          # The number of images to generate(1 ~ 10)
                size = size_type[size]  # The size of the generated images(256x256, 512x512, 1024x1024)
            )

            for data in response['data']:
                images_list.append(data['url'])

            return ["Success", images_list]
    except Exception as e:
        return handle_error(e)