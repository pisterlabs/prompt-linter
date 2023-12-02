#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: anthropic_utils.py
# Author: Zhou
# Date: 2023/6/14
# Copyright: 2023 Zhou
# License:
# Description: anthropic ai claude

import anthropic

from logs.log import logger


class AnthropicAIService:
    max_tokens = 100000
    token_threshold = 1000

    def __init__(self, api_key: str, model_name: str = 'claude-2', **kwargs):
        self.model_name = model_name
        self.claude = anthropic.AsyncAnthropic(api_key=api_key)
        self.client = anthropic.Anthropic()

    def solve_context_limit(self, dialogs: list) -> list:
        """
        reduce the long context to a short one
        :param dialogs: [{'user':"", 'assistant':""}]
        :return: dialogs list
        """
        lgs = [self.client.count_tokens(d['user']) + self.client.count_tokens(
            d['assistant']) for d in dialogs]
        if sum(lgs) > self.max_tokens:
            count = 0
            total = 0
            for num in lgs:
                total += num
                count += 1
                if total > self.max_tokens:
                    break
            dialogs = dialogs[count:]
        return dialogs

    def _generate_msg(self, message, dialog_messages, prompt):
        """
        Generate messages for claude
        :param message:
        :param dialog_messages:
        :param chat_mode:
        """
        if not dialog_messages:
            context = f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"
            return context
        dialog_messages = self.solve_context_limit(dialog_messages)
        context = ''.join(
            [f"{anthropic.HUMAN_PROMPT} {msg['user']} {anthropic.AI_PROMPT} \
            {msg['assistant']}" for msg in dialog_messages])

        context += f"{anthropic.HUMAN_PROMPT} {message} {anthropic.AI_PROMPT}"
        return context

    async def send_message(self, message, dialog_messages=None, prompt=None):
        """
        Send message to claude without stream response
        """
        if dialog_messages is None:
            dialog_messages = []

        try:
            messages = self._generate_msg(message, dialog_messages, prompt)
            resp = await self.claude.completions.create(
                prompt=messages,
                model=self.model_name,
                max_tokens_to_sample=self.token_threshold,
            )
            answer = resp.completion
        except Exception as e:
            logger.error(f"error:\n\n ask: {message} \n with error {e}")
            answer = f"sth wrong with claude, please try again later."

        return answer

    async def send_message_stream(self, message, dialog_messages=None,
                                  prompt=None):
        """
        Send message with stream response
        """
        if dialog_messages is None:
            dialog_messages = []

        try:
            messages = self._generate_msg(message, dialog_messages, prompt)
            answer = await self.claude.completions.create(
                prompt=messages,
                model=self.model_name,
                max_tokens_to_sample=self.token_threshold,
                stream=True
            )
        except Exception as e:
            logger.error(f"error:\n\n ask: {message} \n with error {e}")

            # 创建一个空的异步生成器
            async def empty_generator():
                if False:  # 这将永远不会执行
                    yield

            answer = empty_generator()
        return answer
