#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 基于 threading 实现的异步 request

from time import sleep
import threading
import requests
import langchain_llm


def fetch(url, callback):
    res = requests.get(url)
    sleep(1)
    callback(res.text)


def fetch_async(url, callback):
    fetch_thread = threading.Thread(target=fetch, args=(url, callback))
    fetch_thread.start()

def response_handler(res):
    print(res)

if __name__ == "__main__":

    print(langchain_llm.OpenAI)

    exit()
    print('ok1')
    fetch_async("http://www.baidu.com", response_handler)
    print('ok2')

