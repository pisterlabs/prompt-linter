#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@file: wrapper.py
@time: 2023-03-02 13:17:14.585568
@project: chat_gpt_api_official
@project: ./
"""

import sys
import hashlib
try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls

from aiges.sdk import WrapperBase, \
    StringParamField, \
    ImageBodyField, \
    StringBodyField
from aiges.utils.log import log

########
# 请在此区域导入您的依赖库

# Todo
# for example: import numpy
import openai

########

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
    params1 = StringParamField(key="p1", enums=["3", "eee"], value='3')
    params2 = StringParamField(key="p2", maxLength=44, required=True)
    params3 = StringParamField(key="p3", maxLength=44, required=False)

    # imagebodyfield 指明path，有助于本地调试wrapper.py
    input1 = ImageBodyField(key="data", path="test_data/test.png")
    input3 = ImageBodyField(key="data2", path="test_data/test.png")
    # stringbodyfiled 指明 value，用于本地调试时的测试值
    input2 = StringBodyField(key="switch", value="ctrl")


'''
定义响应类:
 accepts:  accepts代表响应中包含哪些字段, 以及数据类型

 input:    input字段多用与请求数据段，即body部分，当前支持 ImageBodyField, StringBodyField, 和AudioBodyField
'''


class UserResponse(object):
    # 此类定义响应返回数据段，请务必指明对应key
    # 支持 ImageBodyField， AudioBodyField,  StringBodyField
    # 如果响应是json， 请使用StringBodyField
    accept1 = StringBodyField(key="boxes")
    accept2 = StringBodyField(key="boxes2")


'''
用户实现， 名称必须为Wrapper, 必须继承SDK中的 WrapperBase类
'''


class Wrapper(WrapperBase):
    serviceId = "chatgpt"
    version = "v1"
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

    def wrapperInit(cls, config: {}) -> int:
        log.info(config)
        log.info("Initializing ...")
        # doc is here https://platform.openai.com/docs/guides/chat/chat-vs-completions?utm_medium=email&_hsmi=248334739&utm_content=248334739&utm_source=hs_email
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                # system message first, it helps set the behavior of the assistant
                {"role": "system", "content": "You are a helpful assistant."},
                # I am the user, and this is my prompt
                {"role": "user", "content": "What's the best star wars movie?"},
                # we can also add the previous conversation
                # {"role": "assistant", "content": "Episode III."},
            ],
        )
        return 0

    '''
    非会话模式计算接口,对应oneShot请求,可能存在并发调用
    @param params 功能参数
    @param  reqData     请求数据实体字段 DataListCls,可通过 aiges.dto.DataListCls查看
    @return 
        响应必须返回 Response类，非Response类将会引起未知错误
    '''

    def wrapperOnceExec(cls, params: {}, reqData: DataListCls) -> Response:
        log.info("got reqdata , %s" % reqData.list)
        #        print(type(reqData.list[0].data))
        #        print(type(reqData.list[0].data))
        #        print(reqData.list[0].len)
        for req in reqData.list:
            log.info("reqData key: %s , size is %d" % (req.key, len(req.data)))
        log.warning("reqData bytes md5sum is %s" % hashlib.md5(reqData.list[0].data).hexdigest())
        log.info("I am infer logic...please inplement")
        log.info("Testing reqData get: ")
        rg = reqData.get("data")
        log.info("get key: %s" % rg.key)
        log.info("get key: %d" % len(rg.data))

        # test not reqdata
        k = "dd"
        n = reqData.get(k)
        if not n:
            log.error("reqData not has this key %s" % k)

        log.warning("reqData bytes md5sum is %s" % hashlib.md5(reqData.list[0].data).hexdigest())
        log.info("I am infer logic...please inplement")
        r = Response()
        # 错误处理
        # return r.response_err(100)
        l = ResponseData()
        l.key = "ccc"
        l.status = 1
        d = open("test_data/test.png", "rb").read()
        l.len = len(d)
        l.data = d
        l.type = 0
        r.list = [l, l, l]
        return r

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
        r = Response()
        l = ResponseData()
        l.key = "ccc"
        l.status = 1
        d = open("pybind11/docs/pybind11-logo.png", "rb").read()
        l.len = len(d)
        l.data = d
        r.list = [l, l, l]
        return r


if __name__ == '__main__':
    m = Wrapper()
    # m.schema()
    m.run()