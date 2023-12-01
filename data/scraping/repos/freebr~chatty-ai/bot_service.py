import openai
import re
import time
import traceback
import uuid
from importlib import import_module
from logging import getLogger, Logger
from os import path
from revChatGPT.V1 import Chatbot

from .feature.utils.json_parser import parse_json
from .feature.memory import get_memory
from .feature.memory.base import MemoryProviderSingleton
from .feature.utils.command_executor import CommandExecutor
from configure import Config
from definition.cls import Singleton
from definition.const import \
    DIR_CONFIG, COUNT_RECENT_MESSAGES_TO_TAKE_IN, COUNT_RELEVANT_MEMORY_TO_TAKE_IN,\
    MODEL_CHAT, MODEL_MODERATION, MODEL_TEXT_COMPLETION,\
    MAX_TOKEN_CONTEXT, MAX_TOKEN_OUTPUT, MAX_TOKEN_CONTEXT_WITHOUT_HISTORY, REGEXP_TEXT_IMAGE_CREATED, REGEXP_TEXT_SORRY
from handler import msg_handler
from helper.formatter import format_messages, make_message
from helper.token_counter import count_message_tokens, count_string_tokens
from manager import autoreply_mgr, key_token_mgr, user_mgr

URL_OPENAI_API_BASE = 'https://api.openai.com'
MAX_API_INVOKE_COUNT = {
    'api_key': 5,
    'access_token': 1,
}
MAX_OPENAI_COMPLETION_ATTEMPT_NUM = 3
MAX_OPENAI_IMAGE_ATTEMPT_NUM = 3
MAX_OPENAI_SINGLE_ATTEMPT_NUM = 3
MAX_CHAT_FALLBACK_ATTEMPT_NUM = 3
MIN_MESSAGE_HANDLE_LENGTH = 80

getLogger("openai").disabled = True
cfg = Config()
cmd_executor = CommandExecutor()
class BotService(metaclass=Singleton):
    chat_param: dict = {
        'temperature': 0.7,
        'frequency_penalty': 0,
        'presence_penalty': 0,
    }
    chatbots: dict = {}
    key_tokens: dict = {}
    memory: MemoryProviderSingleton
    prompt_files = {
        'free': path.join(DIR_CONFIG, 'prompt-free.txt'),
        'vip': path.join(DIR_CONFIG, 'prompt-vip.txt'),
    }
    preamble_prompt: dict = {}
    services: dict = {}
    logger: Logger
    def __init__(self, **kwargs):
        self.logger = getLogger(self.__class__.__name__)
        self.load_preamble()
        self.update_access_tokens(key_token_mgr.access_tokens.get('Services'))
        self.update_api_keys(key_token_mgr.api_keys.get('Services'))
        self.import_services()
        self.memory = get_memory(cfg)

    def load_preamble(self):
        for prompt_type, prompt_file in self.prompt_files.items():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.preamble_prompt[prompt_type] = f.read()
    
    def import_services(self):
        module = import_module('service', '')
        services = {}
        try:
            api_keys = self.key_tokens.get('api_key')
            if not api_keys: raise Exception('没有可用的 API Key，不能加载服务')
            for class_name, NewService in module.__dict__.items():
                if not str(NewService).startswith("<class 'service"): continue
                services[class_name] = NewService(
                    api_key=api_keys.get(class_name, []),
                    semantic_parse=self.invoke_single_completion,
                )
                self.logger.info('加载服务[%s]成功', class_name)
            self.logger.info('加载服务成功，数量：%d', len(services))
        except Exception as e:
            self.logger.error('加载服务失败：%s', e)
            return False
        self.services = services
        return True

    def update_access_tokens(self, d: dict):
        try:
            access_tokens = self.key_tokens['access_token'] = {}
            for service_name, keys in d.items():
                if service_name == 'OpenAI':
                    dict_openai = access_tokens['OpenAI'] = {}
                    for key in keys:
                        dict_openai[key] = { 'invoke_count': 0 }
                else:
                    if isinstance(keys, str): keys = [keys]
                    access_tokens[service_name] = keys
            if access_tokens == {}: self.logger.warn('没有可用的 Access Token，后备对话服务不可用')
            return True
        except Exception as e:
            self.logger.error(e)
            return False

    def update_api_keys(self, d: dict):
        try:
            api_keys = self.key_tokens['api_key'] = {}
            for service_name, keys in d.items():
                if service_name == 'OpenAI':
                    dict_openai = api_keys['OpenAI'] = {}
                    for key in keys:
                        if key.startswith('sk-'): dict_openai[key] = { 'invoke_count': 0 }
                else:
                    if isinstance(keys, str): keys = [keys]
                    api_keys[service_name] = keys
            if api_keys == {}: self.logger.warn('没有可用的 API Key，对话服务不可用')
            return True
        except Exception as e:
            self.logger.error(e)
            return False

    def begin_invoke(self, type):
        """
        返回一个可用的 Key/Token，调用次数加一
        """
        if self.key_tokens[type] == {}: return
        keys = self.key_tokens[type].get('OpenAI', {})
        if keys == {}: return
        api_key: str = ''
        for key, info in keys.items():
            if info['invoke_count'] >= MAX_API_INVOKE_COUNT.get(type, 1): continue
            info['invoke_count'] += 1
            api_key = key
            break
        if not api_key: api_key = list(keys)[0]
        # 为中间过程调用 OpenAI 接口（如 embedding）指定全局 API Key
        openai.api_key = api_key
        return api_key

    def end_invoke(self, type, value):
        """
        指定的 Key/Token 的调用次数减一
        """
        if self.key_tokens[type] == {}: return
        keys = self.key_tokens[type].get('OpenAI', {})
        if keys == {}: return
        invoke_count = keys[value].get('invoke_count', 0)
        invoke_count = invoke_count - 1 if invoke_count > 0 else 0
        keys[value]['invoke_count'] = invoke_count

    def moderate(self, user, content, api_key=None):
        # 内容审查
        openid = user.get('openid', 'default')
        try:
            new_api_key = api_key if api_key else self.begin_invoke('api_key')
            response = openai.Moderation.create(
                model=MODEL_MODERATION,
                input=content,
                api_key=new_api_key,
            )
            categories: dict = response['results'][0]['categories']
            excluded_categories = ['self-harm']
            result = True, None
            for category, value in categories.items():
                if value and category not in excluded_categories:
                    self.logger.warn('用户 %s 输入的内容被审查分类为[%s]', openid, category)
                    result = False, category
                    break
        except Exception as e:
            self.logger.error(e)
            result = False, 'error'
        finally:
            if not api_key: self.end_invoke('api_key', new_api_key)
        return result

    def make_system_context(self, relevant_memory, records, prompt_type, model):
        """
        生成系统上下文提示
        """
        current_context = [
            # 角色、规则和约束提示
            make_message("system", self.preamble_prompt[prompt_type]),
            # 时间日期提示
            make_message("system", f"当前时间:北京时间{time.strftime('%Y-%m-%d %H:%M:%S')}"),
            # 相关记忆提示
            make_message("system", f"记忆:\n{relevant_memory}\n\n")
        ]
        # 要添加到上下文提示中的历史消息位置，按时间倒序依次添加，直到达到输入 token 上限
        next_message_to_add_index = len(records) - 1
        # 历史消息添加到上下文提示中的位置
        insertion_index = len(current_context)
        # 确定系统提示上下文提示的 token 数
        current_tokens_used = count_message_tokens(current_context, model)
        return next_message_to_add_index, current_tokens_used, insertion_index, current_context
    
    def construct_context(self, user: dict, user_input: str):
        """
        根据指定用户的历史记录和当前输入，构造上下文提示
        """
        send_token_limit = MAX_TOKEN_CONTEXT
        preamble_prompt_type = 'vip' if user_mgr.is_vip(user['openid']) else 'free'
        message_user_input = make_message('user', user_input)
        tokens_user_input = message_user_input['__token']
        # 从长期记忆数据库中取出与上下文相关的记忆
        relevant_memory = self.memory.get_relevant(str(user['records'][-COUNT_RECENT_MESSAGES_TO_TAKE_IN:] + [message_user_input]), COUNT_RELEVANT_MEMORY_TO_TAKE_IN)
        self.logger.info('记忆使用情况：%s', self.memory.get_stats())

        next_message_to_add_index, current_tokens_used, insertion_index, current_context = self.make_system_context(
            relevant_memory, user['records'], preamble_prompt_type, MODEL_CHAT
        )

        current_tokens_used += tokens_user_input
        while current_tokens_used > MAX_TOKEN_CONTEXT_WITHOUT_HISTORY:
            if not relevant_memory: return ('exceed-token-limit', current_tokens_used, MAX_TOKEN_CONTEXT_WITHOUT_HISTORY)
            # 若超出系统提示最大 token 数，从最旧的记忆移除
            relevant_memory = relevant_memory[1:]
            next_message_to_add_index, current_tokens_used, insertion_index, current_context = self.make_system_context(
                relevant_memory, user['records'], preamble_prompt_type, MODEL_CHAT
            )
            current_tokens_used += tokens_user_input

        while next_message_to_add_index >= 0:
            # print (f"CURRENT TOKENS USED: {current_tokens_used}")
            message_to_add = user['records'][next_message_to_add_index]
            tokens_to_add = message_to_add['__token']
            if current_tokens_used + tokens_to_add > send_token_limit:
                break
            # 添加一条历史消息
            current_context.insert(insertion_index, user['records'][next_message_to_add_index])

            # 加历史消息 token 数
            current_tokens_used += tokens_to_add

            # 移动到下一条历史消息位置
            next_message_to_add_index -= 1

        # 添加用户输入消息
        current_context.append(message_user_input)
        # 剩余可用于回答的 token 数
        tokens_remaining = MAX_TOKEN_OUTPUT - current_tokens_used
        assert tokens_remaining >= 0
        return current_context

    def invoke_chat(self, user: dict, user_input: str, is_websocket=False):
        """
        调用 OpenAI API 接口取得问题回答并迭代返回
        """
        api_key = self.begin_invoke('api_key')
        # 过滤敏感词
        user_input = msg_handler.filter_sensitive(user_input)
        # 内容审查
        moderated, category = self.moderate(user, user_input, api_key)
        if not moderated:
            self.end_invoke('api_key', api_key)
            if category == 'error':
                yield make_message('assistant', '')
            else:
                yield make_message('assistant', autoreply_mgr.get('ChatModerationFailed'))
            return
        # 调用文本生成接口
        answer = False
        command_result = ''
        post_prompts = []
        loop_count = 0
        MAX_LOOP_COUNT = 5
        while not answer and loop_count < MAX_LOOP_COUNT:
            # 构造上下文提示
            context = self.construct_context(user, user_input)
            if type(context) == tuple and context[0] == 'exceed-token-limit':
                yield make_message('system', context)
                return
            if post_prompts: context.extend(post_prompts)
            post_prompts.clear()
            assistant_reply = ''
            command_result = ''
            memory_to_add = ''
            last_pos = 0
            loop_count += 1
            # context = [系统提示, 与历史消息有关的记忆, 历史消息, 用户输入]
            for reply in self.invoke_chat_completion_openai(user, context, api_key, is_websocket):
                if reply == 'exceed-token-limit':
                    # TODO: crop messages
                    break
                assistant_reply += reply
                if not answer:
                    if len(assistant_reply) >= 11 and not assistant_reply.startswith('{"command":'):
                        # 开始回答
                        answer = True
                if answer:
                    # 输出回答
                    reply = assistant_reply[last_pos:]
                    last_pos += len(reply)
                    yield make_message('assistant', reply)
            if answer:
                # 保存对话到记忆
                memory_to_add = f"用户输入:{user_input}"\
                    f"\n回复:{assistant_reply}"
                self.save_to_memory(memory_to_add)
            elif assistant_reply:
                self.logger.info(assistant_reply)
                cmds = parse_json(assistant_reply)
                if 'failed' in cmds:
                    # 解析回答失败，直接返回回复
                    self.logger.warn('命令解析失败：%s', assistant_reply)
                    yield make_message('assistant', assistant_reply)
                    answer = True
                    memory_to_add = f"用户输入:{user_input}"\
                        f"\n回复:{assistant_reply}"
                    self.save_to_memory(memory_to_add)
                else:
                    if type(cmds) == dict: cmds = [cmds]
                    for cmd in cmds:
                        # 执行命令
                        command_name, command_result = cmd_executor.exec(cmd['command'], user, self.services)
                        if command_name == '非数学绘画':
                            if command_result == 'no-credit':
                                answer = True
                                yield make_message('assistant', reply)
                                break
                            post_prompts.append(make_message('user', f"""Please translate this into Chinese: Sure, I have drawn an image for you, which was generated by this prompt: "{cmd['command']['args']['prompt']}", what else can I do for you?"""))
                            for url in command_result:
                                reply = f'```image\n![]({url})```'
                                yield make_message('assistant', reply)
                        else:
                            if command_result:
                                if command_result == 'no-credit':
                                    post_prompts.append(make_message('user', f"""Please translate this into Chinese: Sorry, but your 额度 is not enough, so the command "{command_name}" cannot be executed to get the answer to your question. Please consider upgrade your level to gain more 额度."""))
                                    break
                                elif command_result == 'not-supported':
                                    self.logger.warn('调用了不支持的命令：%s', command_name)
                                    post_prompts.append(make_message('user', f"""The command {command_name} is not supported"""))
                                    continue
                                post_prompts.append(make_message('system', """\
    Consider if system has provided information to help you answer the above question, if yes, start answer like:根据我的查询...<give some warmly suggestion>(do not include command JSON)"""))
                            else:
                                # 命令执行没有结果
                                self.logger.warn('命令 %s 执行没有结果：%s', command_name, cmd['command'])
                                continue
                            # 保存中间指令执行结果到记忆
                            memory_to_add = f"执行命令:{command_name}"\
                                f"\n结果:{command_result}"
                            system_message = f'系统查询到以下信息有助于回答用户问题.\n{command_name}结果:{command_result}'
                            user_mgr.add_message(user['openid'], make_message('system', system_message))
                            self.save_to_memory(memory_to_add)
            else:
                # 输入和输出 token 数超出限制
                break
        self.end_invoke('api_key', api_key)
        if not answer: yield make_message('assistant', '')
    
    def invoke_chat_completion_openai(self, user: dict, messages: list, api_key: str, is_websocket=False):
        """
        调用 OpenAI API 接口取得问题回答并迭代返回
        """
        attempt_num = 0
        start = time.time()
        while attempt_num < MAX_OPENAI_COMPLETION_ATTEMPT_NUM:
            try:
                attempt_num += 1
                last_pos = 0
                response = ''
                whole_message = ''
                code_mode = False
                self.logger.info('消息数量：%d', len(messages))
                response = openai.ChatCompletion.create(
                    model=MODEL_CHAT,
                    messages=format_messages(messages),
                    request_timeout=20,
                    stream=True,
                    api_base=f'{URL_OPENAI_API_BASE}/v1',
                    api_key=api_key,
                    temperature=self.chat_param['temperature'],
                    frequency_penalty=self.chat_param['frequency_penalty'],
                    presence_penalty=self.chat_param['presence_penalty'],
                )
                if is_websocket:
                    for res in response:
                        delta = res['choices'][0]['delta']
                        if 'content' not in delta: continue
                        message = delta['content']
                        if message == '\n\n' and not whole_message: continue
                        if res['choices'][0]['finish_reason'] == 'stop': break
                        yield message
                else:
                    task_complete_cmd = False
                    for res in response:
                        delta = res['choices'][0]['delta']
                        if 'content' not in delta: continue
                        text = delta['content']
                        if text == '\n\n' and not whole_message: continue
                        if res['choices'][0]['finish_reason'] == 'stop': break
                        whole_message += text
                        if not task_complete_cmd:
                            if '"command":' not in whole_message: task_complete_cmd = True
                        if task_complete_cmd:
                            if len(whole_message) < MIN_MESSAGE_HANDLE_LENGTH: continue
                            message, last_pos, code_mode = msg_handler.extract_message(
                                text=whole_message[last_pos:],
                                offset=last_pos,
                                min_len=MIN_MESSAGE_HANDLE_LENGTH,
                                code_mode=code_mode,
                            )
                            if len(message) == 0: continue
                            message = msg_handler.filter_sensitive(message)
                            yield message
                    if last_pos == 0:
                        message = msg_handler.filter_sensitive(whole_message)
                        yield message
                    elif last_pos < len(whole_message):
                        message = msg_handler.filter_sensitive(whole_message[last_pos:])
                        yield message
                response_time = time.time() - start
                self.logger.info('响应时间：%ds', response_time)
                return
            except Exception as e:
                if 'This model\'s maximum context length is 4097 tokens.' in str(e):
                    # 裁剪对话
                    attempt_num = 0
                    yield 'exceed-token-limit'
                    return
                else:
                    self.logger.error(e)
                    traceback.print_exc(limit=5)
                    continue
        if attempt_num == MAX_OPENAI_COMPLETION_ATTEMPT_NUM:
            for message in self.invoke_chat_completion_fallback(user, messages, is_websocket):
                yield message

    def invoke_chat_completion_fallback(self, user: dict, messages: list, is_websocket=False):
        """
        调用 revChatGpt 模块取得问题回答并迭代返回
        """
        openid = user.get('openid', 'default')
        conversation_id = user.get('conversation_id')
        parent_id = user.get('parent_id')
        if conversation_id is None:
            conversation_id = uuid.uuid3(uuid.uuid4(), openid + '-conversation')
        if parent_id is None:
            parent_id = uuid.uuid3(uuid.uuid4(), openid + '-conversation-parent')
        self.logger.info('调用 fallback 模块 revChatGpt')
        attempt_num = 0
        access_token = self.begin_invoke('access_token')
        self.logger.info('token: %s', access_token)
        while attempt_num < MAX_CHAT_FALLBACK_ATTEMPT_NUM:
            try:
                attempt_num += 1
                chatbot = self.chatbots[openid] = self.chatbots[openid] if openid in self.chatbots else Chatbot(
                    config={
                        'access_token': access_token,
                        'conversation_id': conversation_id,
                        'parent_id': parent_id,
                    })
                last_pos = 0
                prompt = '\n'.join(['{} says:{}'.format(message['role'], message['content']) for message in messages])
                response = ''
                whole_message = ''
                code_mode = False
                self.logger.info('消息数量：%d', len(messages))
                if is_websocket:
                    for data in chatbot.ask(prompt):
                        conversation_id = data['conversation_id']
                        parent_id = data['parent_id']
                        whole_message = data['message']
                        message = whole_message[last_pos:]
                        last_pos += len(message)
                        if not message: continue
                        yield message
                else:
                    for data in chatbot.ask(prompt):
                        conversation_id = data['conversation_id']
                        parent_id = data['parent_id']
                        whole_message = data['message']
                        response = whole_message[last_pos:]
                        if len(response) < MIN_MESSAGE_HANDLE_LENGTH: continue
                        message, last_pos, code_mode = msg_handler.extract_message(
                            text=response,
                            offset=last_pos,
                            min_len=MIN_MESSAGE_HANDLE_LENGTH,
                            code_mode=code_mode,
                        )
                        if len(message) == 0: continue
                        message = msg_handler.filter_sensitive(message)
                        yield message
                    if last_pos == 0:
                        message = msg_handler.filter_sensitive(response)
                        yield message
                    elif last_pos < len(whole_message):
                        message = msg_handler.filter_sensitive(whole_message[last_pos:])
                        yield message
                self.end_invoke('access_token', access_token)
                user['conversation_id'] = conversation_id
                user['parent_id'] = parent_id
                return
            except Exception as e:
                if 'The message you submitted was too long' in str(e):
                    # 裁剪对话
                    attempt_num = 0
                    messages.pop(1)
                else:
                    self.logger.error(e)
                    traceback.print_exc(limit=5)
                continue
        if attempt_num == MAX_CHAT_FALLBACK_ATTEMPT_NUM:
            self.logger.error('[revChatGPT]尝试 %d 次均无法完成与模型接口的通信，接口调用失败', attempt_num)
        yield ''
        self.end_invoke('access_token', access_token)

    def invoke_single_completion(self, system_prompt='', content=''):
        """
        调用 OpenAI API 接口取得文本填空结果并返回
        """
        attempt_num = 0
        api_key = self.begin_invoke('api_key')
        prompt = ''
        if system_prompt:
            prompt += msg_handler.filter_sensitive(system_prompt) + ':'
        if content:
            prompt += msg_handler.filter_sensitive(content)
        tokens_prompt = count_string_tokens(prompt, MODEL_TEXT_COMPLETION)
        while attempt_num < MAX_OPENAI_SINGLE_ATTEMPT_NUM:
            try:
                attempt_num += 1
                response = openai.Completion.create(
                    model=MODEL_TEXT_COMPLETION,
                    prompt=prompt,
                    request_timeout=20,
                    api_base=f'{URL_OPENAI_API_BASE}/v1',
                    api_key=api_key,
                    max_tokens=MAX_TOKEN_OUTPUT - tokens_prompt,
                    temperature=0,
                )
                if 'text' not in response['choices'][0]: continue
                text = response['choices'][0]['text'].strip()
                self.end_invoke('api_key', api_key)
                return text
            except Exception as e:
                self.logger.error(e)
                continue
        self.end_invoke('api_key', api_key)
        if attempt_num == MAX_OPENAI_SINGLE_ATTEMPT_NUM:
            self.logger.error('[OpenAI API]尝试 %d 次均无法完成与模型接口的通信，接口调用失败', attempt_num)
        messages = [
            make_message('system', system_prompt),
            make_message('user', content),
        ]
        reply = ''
        for message in self.invoke_chat_completion_fallback({}, messages):
            reply += message['content']
        return reply

    def get_chat_param(self):
        return self.chat_param

    def set_chat_param(self, **kwargs):
        try:
            params = [('temperature', 0, 1), ('frequency_penalty', 0, 2), ('presence_penalty', 0, 2)]
            for name, min_val, max_val in params:
                value = float(kwargs.get(name) or self.chat_param[name])
                if not min_val <= value <= max_val: return False
                self.chat_param[name] = value
        except Exception as e:
            self.logger.error('设置 Chat 模型参数失败：', str(e))
            return False
        return True

    def save_to_memory(self, content: str):
        if re.search(REGEXP_TEXT_IMAGE_CREATED, content, re.I): return
        if re.search(REGEXP_TEXT_SORRY, content, re.I): return
        self.memory.add(content)