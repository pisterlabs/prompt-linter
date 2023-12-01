import asyncio
import openai

from wechaty import (
    Contact,
    Message,
    Wechaty,
    MessageType,
    ScanStatus
)
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


sys.path.append(os.path.abspath(os.curdir))


class lower_chatGPT:
    def __init__(self):
        self.statement = "你好~在使用此AI前需要声明：\n" \
                         "1.当前对话不涉及任何隐私信息，双方共同确认对话的可公开性，\n" \
                         "2.对话过程中请勿讨论敏感话题，否则需自行承担不当言论可能造成的法律风险。\n" \
                         "3.如不接受请即刻停止对话，继续对话将被视为完全理解并接受上述声明。\n"\
                         "4.你可以和我对话，然后我可以帮你解决问题。\n例如：(1)你给我主人公是谁，主人公干什么的，然后我可以给你写一首诗,(2)帮你查一下资料，比如宫保鸡丁怎么做的"
        self.suit = ";以上述对话为背景，回答'"
        self.memory_len = 8
        self.memory = {}
        self.apikey = "openai api key"

    def get_backgoud(self, id):
        ans = ""
        print(self.memory[id])
        for bk in self.memory[id]:
            ans += bk+','
        ans += self.suit
        return ans

    def dialog_reply(self, text, id):
        backgroud = self.get_backgoud(id)
        if backgroud == None:
            backgroud = ""
        prompt = str(backgroud)+text+"'"
        print(prompt)
        try:
            openai.api_key = self.apikey
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.8,
                max_tokens=2000,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response["choices"][0]["text"].strip()
        except Exception as exc:
            return "ERROR!"

    async def on_message(self, msg: Message):
        if msg.type() == MessageType.MESSAGE_TYPE_AUDIO:
            await msg.say("不好意思啊我现在不方便听语音，可以打字吗")
            return
        if msg.is_self() or msg.type() != MessageType.MESSAGE_TYPE_TEXT:
            return
        talker = msg.talker()
        text = msg.text()
        id = talker.contact_id

        if id not in self.memory.keys():
            self.memory[id] = []
            await talker.say(self.statement)
            return
        text = text.replace(r'\s', "，").replace("#", "号").replace("&", "和")
        reply = self.remember(text, id)
        await talker.say(reply)

    def remember(self, text, id):
        if id not in self.memory.keys():
            self.memory[id] = []
            self.memory[id].append("Q:'" + text + "',")
        else:
            if len(self.memory[id]) > self.memory_len:
                self.memory[id].pop(0)
            self.memory[id].append("Q:'" + str(text) + "'")
        reply = self.dialog_reply(text, id)
        print(reply)
        if len(self.memory[id]) > self.memory_len:
            self.memory[id].pop(0)
        self.memory[id].append("A:'" + str(reply) + "'")
        return reply


async def main():
    os.environ["WECHATY_PUPPET_SERVICE_TOKEN"] = 'your puppet token'
    os.environ['WECHATY_PUPPET'] = 'wechaty-puppet-padlocal'
    os.environ['WECHATY_PUPPET_SERVICE_ENDPOINT'] = '127.0.0.1:8080'
    wechat = Wechaty()
    Bot = lower_chatGPT()
    wechat.on('message', Bot.on_message)

    await wechat.start()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
