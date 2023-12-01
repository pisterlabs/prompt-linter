from khl import Bot, Message
import openai
import yaml
import os

# KOOK机器人
bot = Bot(token='KOOK_BOT_TOKEN')

# ChatGPT
openai.api_key = 'OPENAI_API_KEY'


# /gpt 发送聊天内容 /gpt own 则是专属于单个用户的聊天
@bot.command()
async def gpt(msg: Message, *command):
    # 检查是否使用户专属聊天
    if 'own' in command:
        # 专属聊天拿用户的ID作为文件名
        ChatFile = 'ChatRecord/chat-' + msg.extra['author']['id'] + '.yaml'
        chat = command[1]
    elif command:
        # 不是专属聊天拿频道ID作为文件名
        ChatFile = 'ChatRecord/chat-' + msg.target_id + '.yaml'
        chat = command[0]
    else:
        await msg.reply('消息内容为空')
        return

    # 判断是否存在旧的聊天记录，有则读取，无则新建
    if os.path.exists(ChatFile):
        with open(ChatFile, 'r', encoding='utf-8') as getTalk:
            data = yaml.safe_load(getTalk)
            # 新建的文件获取到的getTal会为空，需要处理一下
            talk = data if data else []
    else:
        # 新建文件
        os.mknod(ChatFile)
        talk = []

    LatestChat = {"role": "user", "content": chat}

    # 存入ChatRecord
    ChatRecord = talk if len(talk) < 20 else talk[-20:]

    # 往聊天记录添加刚收到的消息
    ChatRecord.append(LatestChat)

    # 调用API
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                              messages=ChatRecord)
    # 回复信息
    await msg.reply(completion.choices[0].message.content)

    # 往聊天记录文件里追加最新的消息
    with open(ChatFile, 'a', encoding='utf-8') as writeTalk:
        yaml.dump([LatestChat], writeTalk, allow_unicode=True, default_flow_style=False)


bot.run()
