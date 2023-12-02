from json import dumps as jsonDumps
from typing import List, Tuple, Union, Literal
from openai import OpenAI
from .chat import system_msg, user_msg, assistant_msg, Chat, AKPool


class GroupChat(Chat):

    roles = {
        "苏轼":'宋朝诗人，他的词风格独特，既有儒家的教诲，又有生活的乐趣。',
        "李清照":'宋代著名的女词人，其词句优美，情感真挚。',
        "杜甫":'唐朝著名诗人。',
        "小许":"一个聪明的程序员",
        "小郑":"一个帅气的男人",
        "小张":"一个漂亮的女人"
    }

    _top_messages: List[dict] = None

    def __init__(self, *vs, **kvs):
        Chat.__init__(self, *vs, **kvs)
        self.roles = self.roles.copy()
        self._new_dialogs = []
        if not self.__class__._top_messages:
            self.__class__._top_messages = self._render_top_messages()
    
    # 欺骗编辑器进行代码提示
    if len(__file__) < 0:
        def __init__(self,
                    # kwargs
                    api_key: Union[str, AKPool],
                    base_url: str = None,  # base_url 参数用于修改基础URL
                    timeout=None,
                    max_retries=None,
                    http_client=None,
                    # request_kwargs
                    model: Literal["gpt-4-1106-preview", "gpt-4-vision-preview", "gpt-4", "gpt-4-0314", "gpt-4-0613",
                                    "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613", "gpt-3.5-turbo"] = "gpt-3.5-turbo",
                    # Chat
                    MsgMaxCount=None,
                    # kwargs
                    **kwargs,
                    ):
            ...
    
    def add_dialog(self, speaker:str, audiences:list, remark:str):
        '''
        add_dialog('李白', ['杜甫', '小许'], '你们好呀')
        add_dialog('杜甫', ['李白'], '你好, 你今天写诗了吗?')
        '''
        self.roles.setdefault(speaker, '')
        for x in audiences:
            self.roles.setdefault(x, '')
        self._new_dialogs.append( f"【{speaker}】对【{'、'.join(audiences)}】说：{remark}".replace('\n', ' ') )
    
    def _get_user_text(self, roles: List[Tuple[str, List[str]]]):
        # 补充人物数据
        for speaker, audiences in roles:
            self.roles.setdefault(speaker, '')
            for x in audiences:
                self.roles.setdefault(x, '')
        # 新增的对话记录
        _new_dialogs = [f"{i+1}、{x}" for i, x in enumerate(self._new_dialogs)]
        _new_dialogs = '\n'.join(_new_dialogs)
        # 需要模拟的对话
        roles = [{'speaker':speaker, 'audiences':'、'.join(audiences), 'remark':''} for speaker, audiences in roles]
        roles = jsonDumps(roles, ensure_ascii=False)
        # 用户的发言
        text = f'''以下是新增的对话记录：\n\n{_new_dialogs}\n\n请根据以下JSON文档的内容，模拟角色的视角进行发言，并把发言填充到'remark'字段中：\n\n```json\n{roles}\n```\n\n请把填充完整后的JSON文档发给我，只返回JSON文档即可，勿包含任何其它信息，否则会干扰我的解析。'''
        return text
    
    def request(self, roles: List[Tuple[str, List[str]]], **kwargs):
        '''
        request([
            ('苏轼', ['李清照', '杜甫']),
            ('小明', ['小东']),
        ])
        '''
        messages = [{"role": "user", "content": self._get_user_text(roles)}]
        self.recently_request_data = {
            'api_key': (api_key := self._akpool.fetch_key()),
        }
        completion = OpenAI(api_key=api_key, **self._kwargs).chat.completions.create(**{
            **self._request_kwargs,  # 全局参数
            **kwargs,  # 单次请求的参数覆盖全局参数
            "messages": self._top_messages + list(self._messages) + messages,
            "stream": False,
        })
        answer: str = completion.choices[0].message.content
        self._messages.add_many(*messages, {"role": "assistant", "content": answer})
        self._new_dialogs = []
        return answer
    
    def _render_top_messages(self):
        messages = []
        # 任务提示
        messages.append(system_msg(f'''以下的JSON文档描述了一些人物的角色信息：\n\n```json\n{jsonDumps(self.roles, ensure_ascii=False)}\n```\n\n你需要了解这些人物的信息。user每次都会收集这些人物的对话记录发送给你，并让你站在其中的某些角色的视角上对其它角色进行模拟发言，你每次返回JSON文档即可，勿包含任何其它信息，否则会干扰user的解析。你在发言时可以编造，比如在回答年龄时，可以随意编一个年龄，但要注意上下文逻辑闭环。'''))
        # 第一轮教学
        self.add_dialog('李白', ['杜甫', '小许'], '你们好呀')
        self.add_dialog('杜甫', ['李白'], '你好, 你今天写诗了吗?')
        self.add_dialog('小许', ['李白'], '你好, 你吃饭了吗?')
        messages.append(user_msg(self._get_user_text([('李白',['小许']), ('李白',['杜甫']), ('李白', ['杜甫','小许'])])))
        messages.append(assistant_msg('''[{"speaker": "李白", "audiences": "小许", "remark": "我今天写诗了"}, {"speaker": "李白", "audiences": "杜甫", "remark": "我吃饭了"}, {"speaker": "李白", "audiences": "杜甫、小许", "remark": "你们有什么有趣的事情分享吗?"}]'''))
        self._new_dialogs = []
        # 第二轮教学
        self.add_dialog('小郑', ['小张'], '你是谁?')
        self.add_dialog('小张', ['小郑'], '我叫小张,今年13岁')
        self.add_dialog('小许', ['小郑', '小张'], '你们是哪里人?')
        messages.append(user_msg(self._get_user_text([('小郑',['小张']), ('小郑',['小许']), ('小张', ['小许'])])))
        messages.append(assistant_msg('''[{"speaker": "小郑", "audiences": "小张", "remark": "哦哦, 我比你大1岁"}, {"speaker": "小郑", "audiences": "小许", "remark": "我是河南人"}, {"speaker": "小张", "audiences": "小许", "remark": "不告诉你"}]'''))
        self._new_dialogs = []
        # 第三轮教学
        self.add_dialog('小许', ['小张'], '呵呵,这么神秘呀?')
        self.add_dialog('小许', ['小郑'], '你知道小张是哪里人吗?')
        messages.append(user_msg(self._get_user_text([('小郑',['小许']), ('小张',['小许'])])))
        messages.append(assistant_msg('''[{"speaker": "小郑", "audiences": "小许", "remark": "哈哈, 我也不知道"}, {"speaker": "小张", "audiences": "小许", "remark": "哈哈, 开个玩笑啦, 我是湖北人"}]'''))
        self._new_dialogs = []
        # 导出
        messages = [dict(x) for x in messages]
        return messages