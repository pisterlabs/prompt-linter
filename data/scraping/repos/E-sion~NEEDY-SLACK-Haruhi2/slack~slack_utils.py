import collections
import os
import pickle
import random
import re
import time
from typing import List, Any

from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

from ChatHaruhi import ChatHaruhi

from consts import *
from slack_bolt import App


def is_dm(message) -> bool:
    # Check if the message is a DM by looking at the channel ID
    # chinese :  通过查看频道ID来检查消息是否为DM
    if message['channel'].startswith('D'):
        return True
    return False


def get_random_thinking_message():
    """
    正在输入中
    """
    return random.choice(thinking_thoughts_chinese)


def send_slack_message_and_return_message_id(app, channel, message: str, thread_ts, is_thread):
    """
    在一个群聊里面进行对话，然后获取到对话的线程，再到线程里面进行回复
    https://api.slack.com/messaging/managing#threading
    （有教程真好）
    :param is_thread:
    :param thread_ts:
    :param app:
    :param channel:
    :param message:
    :return:
    """
    if is_thread:
        response = app.client.chat_postMessage(
            channel=channel,
            text=message,
            thread_ts=thread_ts)
    else:
        response = app.client.chat_postMessage(
            channel=channel,
            text=message)
    if response["ok"]:
        message_id = response["message"]["ts"]
        return message_id
    else:
        return "Failed to send message."


def divede_sentences(text: str) -> List[str]:
    """
    将bot的回复进行分割成多段落
    这个功能可以让bot的回复更加自然，而不是一次性回复一大段话
    但是这个功能在slack上面似乎不太好用，暂时pass吧。
    todo 以后再来研究
    :param text:
    :return:
    """
    sentences = re.split('(?<=[？！])', text)
    return [sentence for sentence in sentences if sentence]


def choose_character(character):
    if character == '糖糖':
        return 糖糖, 'data/characters/NEEDY'
    elif character == '亚璃子':
        return 亚璃子, 'data/characters/AoJiao'
    elif character == '与里':
        return
    # todo 添加更多人物


def run(role, user_prompt, system_prompt, callback, db_folder, is_mention=False):
    # 读取key
    load_dotenv()
    os.environ.get("OPENAI_API_KEY")
    history_name = db_folder.split('/')[-1]
    # 读取本地历史对话（如果有）
    if os.path.exists(f'data/chat_history/{history_name}_history.pkl'):
        with open(f'data/chat_history/{history_name}_history.pkl', 'rb') as f:
            all_dialogue_history = pickle.load(f)
            print(f' 本地聊天记忆库：{all_dialogue_history}\n')
    else:
        all_dialogue_history = []

    system_prompt = system_prompt

    chatbot = ChatHaruhi(system_prompt=system_prompt,
                         llm='openai',
                         story_prefix_prompt='## 经典对话记录 \n 以下是该角色的一段经典对话记录，你要学习模仿这这些经典对话记录的说话口吻,然后再以一种更加口语化的语气进行对话。\n'
                                             '## 历史对话 \n 历史对话在随机对话记录的底下，你需要区分经典对话记录和我们的历史对话 \n',
                         story_db=db_folder,
                         verbose=True,
                         callback=callback,
                         )

    # 在对话之前传入过往对话 并且去重
    chatbot.dialogue_history = list(collections.OrderedDict.fromkeys(all_dialogue_history))

    # 进行回复
    chatbot.chat(role=role, text=user_prompt)

    # 添加聊天记录
    all_dialogue_history.append(chatbot.dialogue_history[-1])  # 只添加最后一条记录

    # 将all_dialogue_history里面的内容保存至本地，作为本地聊天数据库
    with open(f'data/chat_history/{history_name}_history.pkl', 'wb+') as f:
        pickle.dump(all_dialogue_history, f)


# 去除回复中的特定符号
def remove_special_characters(text, name):
    # 删除文本中的「和」
    text = text.replace('「', '')
    text = text.replace('」', '')
    if f"{name}:" in text:
        result = text.replace(f'{name}:', '')
    else:
        result = text.replace(f'{name}：', '')
    return result


# todo 自动更换key似乎无法正常使用，换回调用env文件中设定的key值
def try_keys(keys, user_query, prompt):
    for api_key in keys:
        try:
            response = run('阿p', user_query, prompt)
            return response
        except:
            print(f"key: {api_key} 已失效，正在尝试下一个 key...")
            os.environ["OPENAI_API_KEY"] = api_key

    print("所有的 key 都失效了。")
    return None


CHAT_UPDATE_INTERVAL_SEC = 1
load_dotenv()
bot_token = os.environ["SLACK_BOT_TOKEN"]
app = App(token=bot_token)


class SlackStreamingCallbackHandler(BaseCallbackHandler):
    """
    Slack 流式输出
    """
    last_send_time = time.time()
    message = ""

    def __init__(self, channel, ts):
        self.channel = channel
        self.ts = ts
        self.interval = CHAT_UPDATE_INTERVAL_SEC
        # 投稿を更新した累計回数カウンタ
        self.update_count = 0
        load_dotenv()
        self.name = os.environ["CHARACTER"]

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.message += token
        self.message = remove_special_characters(self.message, self.name)
        now = time.time()
        if now - self.last_send_time > self.interval:
            app.client.chat_update(
                channel=self.channel, ts=self.ts, text=f"{self.message}\n\nTyping ⚙️..."
            )
            self.last_send_time = now
            self.update_count += 1

            # update_countが現在の更新間隔X10より多くなるたびに更新間隔を2倍にする
            if self.update_count / 10 > self.interval:
                self.interval = self.interval * 2

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": self.message}},
        ]
        app.client.chat_update(
            channel=self.channel,
            ts=self.ts,
            text=self.message,
            blocks=message_blocks,
        )


if __name__ == '__main__':
    # try_keys(keys, '你好', '你是一个友善的bot')
    run('阿p', '你好', '你是一个友善的bot')
    # get_vectorstore(path='', key='', base='', save_vectorstore_name='')
    # print('测试成功')
