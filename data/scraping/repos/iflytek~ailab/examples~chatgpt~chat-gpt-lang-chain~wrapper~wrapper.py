#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@file: wrapper.py
@time: 2023-02-08 17:50:55.464284
@project: chat-gpt-lang-chain
@project: ./
"""
import datetime
import random
import sys
import hashlib

try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls, SessionCreateResponse, callback  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls, SessionCreateResponse

from aiges.callback import callback
from aiges.sdk import WrapperBase, \
    StringParamField, \
    ImageBodyField, \
    StringBodyField
from aiges.utils.log import getFileLogger
from aiges.core.types import *

log = getFileLogger()
########
# 请在此区域导入您的依赖库

# Todo
# for example: import numpy
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.llms import OpenAIChat
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory
from aiges.concurrent.fixpool import FixedPool
from aiges.worker import Worker, InferClass


class Infer(InferClass):

    def __init__(self, params={}, tag="", sid=None):
        self.tag = tag
        self.sid = sid
        template = """Assistant is a large language model trained by OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

        {history}
        Human: {human_input}
        Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template
        )

        self.model = LLMChain(llm=OpenAIChat(model_name='gpt-3.5-turbo', temperature=0,
                                             openai_api_key=params.get("api_key")),
                              prompt=prompt,
                              verbose=True,
                              memory=ConversationalBufferWindowMemory(k=100),
                              )

    def setHandle(self, hdl):
        log.debug("#### setinghandl %s" % hdl)
        self.handle = hdl

    def predict(self, *args, **kwargs):
        # 实现并回调结果
        log.info(kwargs)
        req = kwargs.get('req').get("message")
        input_text = req.data.decode('utf-8')
        input_status = req.status
        r = random.randint(1, 100000)
        now = datetime.datetime.now().strftime("%y-%m-%d: %H:%M")
        #output = f'iam the mock result for test: {now}  {r}'

        output = self.model.predict(human_input=input_text)
        r = Response()
        d = ResponseData()
        d.setData(output.encode("utf-8"))
        all_done = False
        d.key = "response"
        log.info([input_status])
        if input_status == DataContinue:
            d.status = DataContinue
        elif input_status == DataEnd:
            d.status == DataEnd
            all_done = True

        elif input_status == DataOnce:
            d.status = DataEnd
            all_done = True
        elif input_status ==DataBegin:
            d.status = DataContinue

        r.list = [d]
        log.info("handle callback %s, callback, status: %s" % (self.tag,d.status))
        callback(r, self.tag, self.sid)
        return output, all_done


'''
定义请求类:
 params:  params 开头的属性代表最终HTTP协议中的功能参数parameters部分，
          params Field支持 StringParamField，
          NumberParamField，BooleanParamField,IntegerParamField，每个字段均支持枚举
          params 属性多用于协议中的控制字段，请求body字段不属于params范畴

 input:    input字段多用与请求数据段，即body部分，当前支持 ImageBodyField, StringBodyField, 和AudioBodyField
'''


class UserRequest(object):
    # StringParamField多用于控制参数
    # 指明 enums, maxLength, required有助于自动根据要求配置协议schema
    # params1 = StringParamField(key="message", value=b'')
    # params2 = StringParamField(key="p2", maxLength=44, required=True)
    params3 = StringParamField(key="api_key", maxLength=44, required=True,
                               value=b"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")

    # imagebodyfield 指明path，有助于本地调试wrapper.py
    # input1 = ImageBodyField(key="message", path="test_data/test.png")
    # input3 = ImageBodyField(key="data2", path="test_data/test.png")
    # stringbodyfiled 指明 value，用于本地调试时的测试值
    c = "计算机是由什么组成的?".encode("utf-8")
    input2 = StringBodyField(key="message", value=c)


'''
定义响应类:
 accepts:  accepts代表响应中包含哪些字段, 以及数据类型

 input:    input字段多用与请求数据段，即body部分，当前支持 ImageBodyField, StringBodyField, 和AudioBodyField
'''


class UserResponse(object):
    # 此类定义响应返回数据段，请务必指明对应key
    # 支持 ImageBodyField， AudioBodyField,  StringBodyField
    # 如果响应是json， 请使用StringBodyField
    accept1 = StringBodyField(key="response")


'''
用户实现， 名称必须为Wrapper, 必须继承SDK中的 WrapperBase类
'''


class Wrapper(WrapperBase):
    serviceId = "chatgpt"
    call_type = 1
    version = "backup.0"
    requestCls = UserRequest()
    responseCls = UserResponse()

    '''
    服务初始化
    @param config:
        插件初始化需要的一些配置，字典类型
        key: 配置名
        value: 配置的值
    @return
        ret: 错误码。无错误时返回0
    '''

    def wrapperInit(self, config: {}) -> int:
        log.info(config)
        log.info("Initializing ...")
        lics = 10
        self.pool = FixedPool()
        self.pool.set_capacity(int(config.get("common.lic", lics)))

        return 0

    '''
    非会话模式计算接口,对应oneShot请求,可能存在并发调用
    @param params 功能参数
    @param  reqData     请求数据实体字段 DataListCls,可通过 aiges.dto.DataListCls查看
    @return
        响应必须返回 Response类，非Response类将会引起未知错误
    '''

    def wrapperOnceExec(cls, params: {}, reqData: DataListCls) -> Response:
        pass
        print("#####")
        print(params)
        print(reqData)
        return None

    '''
    服务逆初始化

    @return
        ret:错误码。无错误码时返回0
    '''

    def wrapperFini(cls) -> int:
        return 0

    '''
    非会话模式计算接口,对应oneShot请求,可能存在并发调用
    @param ret wrapperOnceExec返回的response中的error_code 将会被自动传入本函数并通过http响应返回给最终用户
    @return
        str 错误提示会返回在接口响应中
    '''

    def wrapperError(cls, ret: int) -> str:
        if ret == 100:
            return "user error defined here"
        return ""

    '''
        此函数保留测试用，不可删除
    '''

    def wrapperTestFunc(cls, data: [], respData: []):
        pass
        return None

    def wrapperCreate(self, params: {}, sid: str, userTag: str = "") -> SessionCreateResponse:
        """
        非会话模式计算接口,对应oneShot请求,可能存在并发调用
        @param ret wrapperOnceExec返回的response中的error_code 将会被自动传入本函数并通过http响应返回给最终用户
        @return
            SessionCreateResponse类, 如果返回不是该类会报错
        """
        log.info("Getting paramas: %s" % str(params))
        worker = Worker()
        worker.register_infer_class(Infer(params, userTag, sid))
        worker.register_infer_func(self.wrapperOnceExec)
        worker.register_callback(callback)

        ok, handle = self.pool.add(worker)

        sp = SessionCreateResponse()

        if not ok:
            sp.handle = ""
            sp.error_code = 40001
            return sp
        worker.setHandle(handle)
        worker.start()

        # _session.setup_callback_fn(callback)
        sp.handle = handle
        sp.error_code = 0
        return sp

    def wrapperWrite(self, handle: str, req: DataListCls, sid: str) -> int:
        """
        会话模式下: 上行数据写入接口
        :param handle: 会话handle 字符串
        :param req:  请求数据结构
        :param sid:  请求会话ID
        :return:
        """
        _session = self.pool.get(handle)
        # _session = self.session.get_session(handle=handle)
        if _session == None:
            log.info("can't get this handle: %s" % handle)
            return -1
        _session.in_q.put(req)
        return 0

    def wrapperRead(self, handle: str, sid: str) -> Response:
        """
        会话模式: 当前此接口在会话模式且异步取结果时下不会被调用！！！！！返回数据由callback返回
        同步取结果模式时，下行数据返回接口
                  如果为异步返回结果时，需要设置加载器为asyncMode=true [当前默认此模式],
        :param handle: 请求数据结构
        :param sid: 请求会话ID
        :return: Response类
        """
        return None

    def wrapperDestroy(self, handle: str) -> int:
        log.debug("destroying %s" % handle)
        self.pool.remove(handle)
        return 0


if __name__ == '__main__':
    m = Wrapper(legacy=False)
    m.schema()
    #m.run(stream=True)
