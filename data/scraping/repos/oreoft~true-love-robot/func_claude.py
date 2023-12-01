#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import claude_api
import openai
import time


class Claude():

    def __init__(self, config) -> None:
        self.claude_client  = claude_api.Client(config.get("cookie"))
        self.conversation_table = {}

    def get_answer(self, question: str, wxid: str, sender: str) -> str:
        print(question)
        # 先根据发送信息获取uuid
        conversation_id = self.conversation_table.get("wxid", self.claude_client.create_new_chat()['uuid'])
        start_time = time.time()
        print("开始发送给claude， 其中conversation_id: ", conversation_id, " wxid: ", wxid)
        try:
            # 然后根据uuid发送问题
            rsp = self.claude_client.send_message(question, conversation_id)
        except Exception as e0:
            rsp = "发生未知错误：" + str(e0)
        end_time = time.time()
        cost = round(end_time - start_time, 2)
        print("chat回答时间为：", cost, "秒")
        if question.startswith('debug'):
            return rsp + '\n\n' + '(cost: ' + str(cost) + 's, conversation_id: ' + conversation_id[-4:] + ', wxid: ' + wxid + ')'
        else:
            return rsp

if __name__ == "__main__":
    from configuration import Config
    config = Config().CLAUDE
    if not config:
        exit(0)
    chat = Claude(config)
    # 测试程序
    while True:
        q = input(">>> ")
        try:
            time_start = datetime.now()  # 记录开始时间
            print(chat.get_answer(q, "wxid_tqn5yglpe9gj21", "wxid_tqn5yglpe9gj21"))
            time_end = datetime.now()  # 记录结束时间
            print(f"{round((time_end - time_start).total_seconds(), 2)}s")  # 计算的时间差为程序的执行时间，单位为秒/s
        except Exception as e:
            print(e)
