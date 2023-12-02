#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Lkeme
import os
from typing import Union

from app.adapter.message import AdapterMessageSegment
from app.config import config_mg
from app.logger import logger
from app.permission import ROLE

try:
    import openai
except Exception as _:  # noqa
    os.system('pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple')
    import openai

from app.adapter.events import MessageEvent
from app.plugin import plugin_mg as PM
from app.utils.funcs import re_filter
from dataclasses import dataclass
from app.utils.draw import normal_image_draw


@dataclass
class Config:
    # 插件基础信息
    name: str = 'ChatGPT插件'  # 插件名称
    path = 'chatgpt.default'  # 插件导入路径
    description: str = 'ChatGPT插件'  # 插件描述
    version: str = '0.0.1'  # 插件版本
    author: str = 'Lkeme'  # 插件作者
    data: bool = False  # 是否有外部资源
    # 插件配置
    enable: bool = False
    openai_api_key: str = ''
    # 'text-davinci-003'  # engine="text-davinci-002"
    chatgpt_model: str = ''
    chatgpt_max_tokens: int = 500  # 3400
    chatgpt_temperature: float = 0.9

    def __post_init__(self):
        self.enable = config_mg.get_bool(self.path, 'enable', appoint='plugin')
        self.openai_api_key = config_mg.get(self.path, 'openai_api_key', appoint='plugin')
        self.chatgpt_model = config_mg.get(self.path, 'chatgpt_model', appoint='plugin')


CONFIG: Union[Config, None] = None


def init_config():
    """
    初始化配置
    """
    config_mg.add(Config.path, 'enable', 'false', appoint='plugin')
    config_mg.add(Config.path, 'openai_api_key', '', appoint='plugin')
    config_mg.add(Config.path, 'chatgpt_model', 'text-davinci-003', appoint='plugin')
    config_mg.save(appoint='plugin')
    #
    global CONFIG
    CONFIG = Config()


init_config()

# Conversation with AI assistant
# This is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.
preset = ''


@PM.reg_event('message')
@re_filter("^chat (.*?)$", enable=CONFIG.enable)
async def ai_chat(event: MessageEvent) -> None:
    if CONFIG.openai_api_key == '':
        return
    openai.api_key = CONFIG.openai_api_key
    data = str(event.message).replace('chat ', '').strip()
    logger.info(f'openai completion for: {data}')
    try:
        # https://beta.openai.com/docs/api-reference/completions/create
        global preset
        resp = openai.Completion.create(
            model=CONFIG.chatgpt_model,
            prompt=preset + '\n\n' + data,
            max_tokens=CONFIG.chatgpt_max_tokens,
            temperature=CONFIG.chatgpt_temperature,
            request_timeout=30
        )
        msg = resp.choices[0]['text']
        msg = msg.split('\n\n', 1)[-1]
    except Exception as e:
        # await m.send('错误发生：%s' % str(e), at_sender=True)
        # TODO 发表情
        raise e

    logger.info(f'get completion: {msg}')
    try:
        # msg = AdapterMessageSegment.reply(event.message_id) + msg
        msg = str(AdapterMessageSegment.text(msg))
        await event.reply(msg)
    except Exception as e:
        # await m.send('回答已获取，但是发送失败', at_sender=True)
        # TODO 发表情
        raise e


@PM.reg_event('message')
@re_filter("^chat preset (.*?)$", enable=CONFIG.enable)
async def chat_preset(event: MessageEvent) -> None:
    data = str(event.message).replace('chat ', '').strip()
    global preset
    if data:
        preset = data
        logger.success('set preset to: %s' % preset)
        await event.reply('新预设已加载')
    else:
        await event.reply(f'当前预设：\n{preset}')


@PM.reg_event('message')
@re_filter("^chat reset preset$", enable=CONFIG.enable)
async def chat_preset_reset(event: MessageEvent) -> None:
    global preset
    preset = ''
    await event.reply('预设已重置')


@PM.reg_event('message')
@re_filter("^chatgpt\?$", role=ROLE.ADMIN, enable=CONFIG.enable)
async def chatgpt_help(event: MessageEvent):
    plain_text = """chatgpt? - 帮助 \nchat <内容> - 请求答复内容 \n"""
    try:
        pic = await normal_image_draw(plain_text)
        content = AdapterMessageSegment.image(pic)
        await event.reply(str(content))
    except Exception as e:
        logger.error(e)
