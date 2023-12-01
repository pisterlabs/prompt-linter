#!/usr/bin/python3
# -*- coding:utf-8 -*-
import asyncio
import datetime
import os
import uuid

import openai
from fastapi import APIRouter, Depends, Header
from loguru import logger
from starlette.responses import StreamingResponse

from chatgpt.conf.mysettings import settings
from chatgpt.database_option import crud, schemas
from chatgpt.db import db_session, Session
from chatgpt.message_bodys import *
from chatgpt.models.users import SecretKey, User
from chatgpt.schemas.items import VerifySchema
from chatgpt.schemas.users import GenKeySchema, NodeToken
from chatgpt.util import num_tokens_from_messages, get_proxies
from chatgpt.utils.jwt_tool import JwtTool
from chatgpt.utils.secret_utils import SecretUtils

app = APIRouter()


@app.post('/chat-process2')
async def chat_process2(data: ChatProcessRequest, db: Session = Depends(crud.get_db)):
    """
    chatGpt对话请求
    :param data:
    :param db:
    :return:
    """
    chat_response = ChatProcessResponse()
    # 配置对话内容
    messages = []
    if data.options.parentMessageId:
        # 查询历史会话
        db_conversation = crud.get_conversation_by_id(db, id=data.options.parentMessageId)
        if db_conversation:
            messages = db_conversation.contents
    if not messages:
        messages.append({"role": "system", "content": data.systemMessage})
    chat = {"role": "user", "content": data.prompt}
    messages.append(chat)
    # 默认gpt3.5模型
    model = "gpt-3.5-turbo-0301"
    # 计算token
    chat_response.question_token = await num_tokens_from_messages(messages, model=model)
    # openai请求参数
    params = {
        "model": model,
        "messages": messages,
        'temperature': data.temperature,
        'top_p': data.top_p,
        "stream": True
    }
    # 设置代理
    openai.proxy = get_proxies(package="openai")
    # 创建会话
    chat_reply_process = await asyncio.get_running_loop().run_in_executor(None, lambda: openai.ChatCompletion.create(
        **params))

    async def generate():
        for index, chat in enumerate(chat_reply_process):
            detail = chat.to_dict_recursive()
            choice = detail.get("choices")[0]
            delta = choice.get("delta")
            if not chat_response.role:
                chat_response.role = delta.get("role", "")
            if not chat_response.id:
                chat_response.id = detail.get("id", "")
            chat_response.text += delta.get("content", "")
            chat_response.delta = delta
            chat_response.detail = detail
            chat_response.answer_token = await num_tokens_from_messages(
                [{"role": "system", "content": chat_response.text}], model=model)
            response = chat_response.json(ensure_ascii=False)
            yield f"\n{response}" if index else response

        consume_token = [chat_response.question_token, chat_response.answer_token]
        messages.append({"role": chat_response.role, "content": chat_response.text})
        # 更新会话消息
        if data.options.parentMessageId:
            # 若存在则只更新
            try:
                result = crud.update_conversation(db, id=data.options.parentMessageId, messages=messages,
                                                  new_id=chat_response.id, consume_token=consume_token)
                logger.info(
                    f"id:{result.id} title:{result.title} consume_token:{result.consume_token} messages:{result.contents}")
            except Exception as e:
                logger.error(
                    f"Update error: id={data.options.parentMessageId} new_id={chat_response.id} consume_token:{consume_token} messages={messages}")
                logger.error(f"Insert error reason: {e.__str__()}")
        else:
            conversation = schemas.ConversationInsert(
                id=chat_response.id,
                user_id=None,
                title=data.prompt,
                contents=messages,
                create_time=datetime.datetime.now(),
                consume_token=consume_token
            )
            # 插入新的会话
            try:
                result = crud.insert_conversation(db, conversation)
                logger.info(
                    f"id:{result.id} title:{result.title} consume_token:{result.consume_token} messages:{result.contents}")
            except Exception as e:
                logger.error(f"Insert error: {conversation.json(ensure_ascii=False)}")
                logger.error(f"Insert error reason: {e.__str__()}")

    # 流式返回
    return StreamingResponse(content=generate(), media_type="application/octet-stream")


@app.post('/request')
async def request(x_token: str = Header(...)):
    """
    调用openai前查询当前用户是否还有剩余次数
    :param x_token:
    :return:
    """
    try:
        info = JwtTool.check_access_token(x_token)
        if info:
            username = info['username']
            # 查询数据库剩余次数
            with db_session as session:
                rows = session.query(User).filter_by(username=username).with_for_update().limit(1).all()
                # 如果有这个用户，更新次数
                if rows:
                    remaining_count = rows[0].remaining_count
                    if remaining_count > 0:
                        return {'message': 'request处理成功！', 'status': 'success'}
        return {'message': f'request接口异常！', 'status': 'error'}
    except Exception as e:
        logger.error(e)
    return {'message': f'request接口异常！', 'status': 'error'}


@app.post('/charging')
async def charging(x_token: str = Header(...)):
    # 这个接口在api调用成功后再触发，否则api未访问成功也扣费，逻辑有问题
    try:
        info = JwtTool.check_access_token(x_token)
        if info:
            username = info['username']
            # 查询数据库剩余次数
            with db_session as session:
                rows = session.query(User).filter_by(username=username).with_for_update().limit(1).all()
                # 如果有这个用户，更新次数
                if rows:
                    remaining_count = rows[0].remaining_count
                    if remaining_count - 1 < 0:
                        return {'message': '用户剩余次数为0！', 'status': 'error'}
                    else:
                        rows[0].remaining_count -= 1
                        session.commit()
                        return {'message': 'request处理成功！', 'status': 'success'}
                else:
                    # 没有这个用户
                    return {'message': 'request接口异常！', 'status': 'error'}
        return {'message': 'request接口异常！', 'status': 'error'}
    except Exception as e:
        return {'message': f'request接口异常！{e}', 'status': 'error'}


@app.post("/session", response_model=SessionResponse, summary="")
async def session():
    """
    session认证
    :return:
    """
    response = SessionResponse()
    return response


@app.post('/verify')
async def verify(item: VerifySchema):
    """
    验证前端秘钥，他秘钥的名字是token，暂时不改他名称
    :param item:
    :return:
    """
    # 当前白嫖用户使用秘钥进行验证，更新换秘钥重新生成jwt即可
    secret_key = item.token
    try:
        with db_session as session:
            rows = session.query(SecretKey).all()
            normal_key = rows[0].normal_key
            # 2类key,微信群的key
            group_key = rows[0].group_key
            # 3类key,星球的key
            vip_key = rows[0].vip_key
            # 管理员秘钥
            # admin_key = rows[0].admin_key
            if secret_key in [normal_key, str(normal_key), group_key, str(group_key), vip_key, str(vip_key)]:
                num_map = {
                    str(normal_key): settings.NORMAL_NUM,
                    str(group_key): settings.GROUP_NUM,
                    str(vip_key): settings.VIP_NUM,
                    # str(admin_key): 9999999,
                }
                cur_num = num_map[secret_key]
                username = uuid.uuid4()
                # 写入到数据库
                rows = session.query(User).filter_by(username=username).with_for_update().all()
                if not rows:
                    new_object = User(username=username, remaining_count=cur_num)
                    session.add(new_object)
                    session.commit()
                jwt = JwtTool.create_access_token(username=str(username), num=cur_num)
                logger.info(jwt)
                return {'status': 'Success', 'message': f'免费用户:{username}添加成功', 'data': '', 'token': jwt}
            else:
                return {'message': "非法秘钥！", 'status': 'error'}
    except Exception as e:
        return {'message': f'verify接口异常：{e}', 'status': 'error'}


@app.post('/gen_key')
async def gen_key(item: GenKeySchema):
    """
    更新秘钥
    :param item:
    :return:
    """
    if item.admin_token == settings.ADMIN_TOKEN_LIST:
        # 更新数据库中的秘钥
        with db_session as session:
            rows = session.query(SecretKey).with_for_update().limit(1).all()
            normal_key, group_key, vip_key = SecretUtils.gen_secret_key()
            for row in rows:
                row.normal_key = normal_key
                row.group_key = group_key
                row.vip_key = vip_key
            session.commit()
            return {'message': '秘钥更新完成！', 'status': 'success', 'normal_key': normal_key, 'group_key': group_key,
                    'vip_key': vip_key}

    return {'message': '无效的秘钥！', 'status': 'error'}


@app.post('/openai')
async def openai_key(item: NodeToken):
    """
    传递openai的秘钥给node
    :param item:
    :return:
    """
    if item.token not in ['openai']:
        return {'message': '非法token', 'status': 'error'}
    key = os.getenv('OPENAI_KEY')
    return {'message': '获取key', 'status': 'success', 'apiKey': key}
