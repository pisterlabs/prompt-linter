import base64
import json
import logging
import sys
import openai

import tornado.gen
import tornado.web

from utils.responseHelper import ResponseHelper
from utils.utilsFile import UtilsFile


class Handler_updateChat(tornado.web.RequestHandler):

    @tornado.gen.coroutine
    def get(self):
        self.set_header("Content-Type", "text/plain")
        self.finish('copy')

    @tornado.gen.coroutine
    def post(self):
        self.dispose()


    def dispose(self):
        try:
            body = self.request.body
            content = json.loads(body)
            asks = content['asks']
            answers = content['answers']
            new_chat = content['new_chat']
            scenario = content['scenario']

            answer, history_prompts, history_answers = self.chat(asks, answers, new_chat, scenario)

            response = ResponseHelper.generateResponse(True)
            response['answer'] = answer
            response['asks'] = asks
            response['answers'] = answers

            self.write(json.dumps(response))
            self.finish()

        except Exception as e:
            print('server internal error')
            logging.exception(e)

            self.set_header("Content-Type", "text/plain")
            response = ResponseHelper.generateResponse(False)
            self.write(json.dumps(response))
            self.finish()


    def chat2gpt(self, cur_prompt, system, history_prompts, history_answers,
                 max_tokens=1024, n=1, temperature=0.8, stop=None,
                 model="gpt-3.5-turbo"):
        messages = []

        messages.append({
            "role": 'system',
            "content": system,
        })
        messages.append({
            "role": 'user',
            "content": system,
        })

        round = min(len(history_prompts), len(history_answers))
        start = max(0, round - 3)
        for r in range(start, round):
            if r < len(history_prompts):
                messages.append({
                    "role": 'user',
                    "content": history_prompts[r],
                })

            if r < len(history_answers):
                messages.append({
                    "role": 'assistant',
                    "content": history_answers[r],
                })

        messages.append({
            "role": 'user',
            "content": cur_prompt,
        })

        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
        )
        return completion.choices[0].message.content


    def chat(self, history_prompts, history_answers, cur_prompt, scenario):

        system_limit = self.get_system_limit()
        system = '\n'.join([system_limit, scenario])

        # consist history prompt
        '''
        prompt = ""
        _history_prompts = history_prompts[-2:]
        _history_answers = history_answers[-2:]
        for i in range(len(_history_prompts)):
            prompt += _history_prompts[i] + '\n'
            prompt += _history_answers[i] + '\n'
        prompt += cur_prompt
        '''

        answer = self.chat2gpt(cur_prompt, system, history_prompts, history_answers)
        history_prompts.append(cur_prompt)
        history_answers.append(answer)

        return answer, history_prompts, history_answers


    def get_system_limit(self):
        system_limit = '''
        你的回答不能超过 512 个字符.
        '''
        return system_limit


