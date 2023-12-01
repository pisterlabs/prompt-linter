import asyncio
import websockets
import openai
openai.api_key="你的api key"
async def echo(websocket):
    async for message in websocket:
        await websocket.send("Loading")

        if(" "in message):
              response = openai.ChatCompletion.create(
                 model="gpt-3.5-turbo",
                 messages=[
                     {"role": "system",
                     "content": "请将下面的句子翻译成中文，谢谢"},
                    {"role": "user", "content": message},
                 ]
             )
        else:
            response = openai.ChatCompletion.create(
                 model="gpt-3.5-turbo",
                 messages=[
                  {"role": "system",
                      "content": "请给出以下单词的词性，详细的意思，用法，以及例句，谢谢"},
                  {"role": "user", "content": message},
               ]
          )
        await websocket.send(response['choices'][0]['message']['content'])
async def main():
    async with websockets.serve(echo, "你的服务器ip", 8765):
        await asyncio.Future()  # run forever
asyncio.run(main())
