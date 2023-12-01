import asyncio
import json

import aiohttp
from pyrogram import Client
from pyrogram.types import Message

from core import command
from tools.helpers import Parameters
from pyrogram.enums import ParseMode

from core import bot_config

# import openai

fakeopen_completions_url = 'https://ai.fakeopen.com/v1/chat/completions'


def get_access_token() -> str:
    return bot_config['chatgpt'].get('access_token')


async def fakeopen_completions_chat(query: str, stream_true: bool):
    params = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": f"{query}"
            }
        ]
        ,
        'stream': stream_true,
    }
    json_data = json.dumps(params)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + get_access_token(),
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/118.0.0.0 Safari/537.36'
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(fakeopen_completions_url, data=json_data, headers=headers) as response:
            if response.status == 200:
                # 使用iter_any()方法逐块读取流式数据
                result = ""
                async for chunk in response.content.iter_any():
                    # 在这里处理每个数据块
                    print(chunk.decode("utf-8"))
                    decoded_chunk = chunk.decode("utf-8")
                    if 'choices' in decoded_chunk and 'delta' in decoded_chunk['choices'][0]:
                        chunk_msg = decoded_chunk['choices'][0]['delta'].get('content', '')
                        result += chunk_msg  # 将输出内容附加到结果字符串上

                        if stream_true:
                            # print(chunk_msg, end='', flush=True)
                            await asyncio.sleep(0.05)
                print(result)
                return result

            else:
                print(f"Failed to fetch data. Status code: {response.status}")
    return ""


# def fakeopen_api(query: str, max: int, stream_true: bool, tem: float):
#     openai.api_base = fakeopen_base
#     openai.api_key = get_access_token()
#     # start_time = time.time()  # 记录开始时间
#
#     response = openai.ChatCompletion.create(
#         model='gpt-3.5-turbo',
#         messages=[
#             {'role': 'user', 'content': query}
#         ],
#         temperature=tem,
#         max_tokens=max,
#         stream=True  # 开启流式输出
#     )
#
#     result = ""  # 创建一个空字符串来保存流式输出的结果
#
#     for chunk in response:
#         # 确保字段存在
#         if 'choices' in chunk and 'delta' in chunk['choices'][0]:
#             chunk_msg = chunk['choices'][0]['delta'].get('content', '')
#             result += chunk_msg  # 将输出内容附加到结果字符串上
#
#             if stream_true:
#                 # print(chunk_msg, end='', flush=True)
#                 time.sleep(0.05)
#
#     return result  # 返回流式输出的完整结果


@Client.on_message(command('chatgpt'))
async def chatgpt(client: Client, message: Message):
    """与chatgpt对话"""
    cmd, args = Parameters.get(message)
    query_str = args
    await message.edit_text("🌍正在询问chatgpt,请稍后...")
    full_result = await fakeopen_completions_chat(query_str, True)
    if not full_result:
        full_result = "无法获取chatgpt回复,请检查插件是否正常工作."
    result_text = f"🔎 | **ChatGPT** | `回复`\n{full_result}"
    await message.edit_text(
        text=result_text,
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True
    )


if __name__ == '__main__':
    asyncio.run(fakeopen_completions_chat("你好", True))
