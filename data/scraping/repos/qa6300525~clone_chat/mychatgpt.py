#!/usr/bin/env python
# coding: utf-8
from uuid import uuid4

from flask import Blueprint, request, render_template, flash, redirect, url_for

from flask import g, jsonify
from threading import Lock

from flask_login import login_required, current_user
from app import chat_histories
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


chat = Blueprint('chat', __name__)


@chat.before_request
def before_request():
    if 'locks' not in g:
        g.locks = {}


@chat.route('/chat/get_msg', methods=['GET', 'POST'])
@login_required
def get_msg():
    # 登录校验
    if current_user.is_authenticated is False:
        return redirect(url_for('auth.login'))
    # 剩余次数校验
    if current_user.remaining_count <= -10000:
        return jsonify({
            'message': f"你当前剩余次数为{current_user.remaining_count}，请购买更多次数。"
        })
    email = current_user.email
    if email not in g.locks:
        g.locks[email] = Lock()

    lock = g.locks[email]

    if lock.locked():
        return {"error": "Another request is being processed."}, 429

    # Add lock to prevent multiple requests from the same user.
    with lock:
        bot_type_str = 'gpt-3.5-turbo'
        message = request.json.get('message')
        session_id = request.json.get('session_id')

        # If session_id is not provided or not found in chat_histories,
        # create a new session.
        if not session_id or session_id not in chat_histories:
            session_id = str(uuid4())
            chat_histories[session_id] = []
        # Add user's message to chat history.
        chat_histories[session_id].append(HumanMessage(content=message))

        # TODO: Send message to langchain API and get model's reply.
        # reply = ...
        msgs = chat_histories[session_id]
        chat_ai = ChatOpenAI(temperature=0.5,
                             openai_api_key=os.getenv("OPENAI_API_KEY"),
                             openai_api_base='https://api.openai-proxy.com/v1')
        reply = chat_ai(msgs)
        print(reply.content)
        current_user.remaining_count -= 1
        current_user.update_remaining_count(current_user.remaining_count)
        print(current_user.remaining_count)
        # Add model's reply to chat history.
        chat_histories[session_id].append(AIMessage(content=reply.content))

    return jsonify({
        'session_id': session_id,
        'message': reply.content
    })
