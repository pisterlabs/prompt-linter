#!/usr/bin/env python3
# encoding: utf-8

from typing import Any, List, Dict, Mapping, Optional
from langchain.llms.base import LLM
from langchain.llms import OpenAI
import requests
import re
import vim
import json
import traceback
import time

def contains_nr(s):
    ascii_list = [ord(c) for c in s]
    return 10 in ascii_list or 13 in ascii_list

def is_all_nr(s):
    if s.count("\n") == len(s) or s.count("\r") == len(s):
        return True
    else:
        return False

def vim_command_handler(script):
    if script == "[DONE]":
        vim.command("call nvim_ai#teardown()")
        vim.command("echom '[DONE]'")
        return
    # elif script == "\n":
    #     vim.command("call nvim_ai#new_line()")
    elif is_all_nr(script):
        for i in range(len(script)):
            vim.command("call nvim_ai#new_line()")
    elif contains_nr(script):
        script_items= script.split("\n")
        tmp_count = 0
        for item in script_items:
            if item == "":
                vim.command("call nvim_ai#new_line()")
            else:
                if tmp_count > 0:
                    vim.command("call nvim_ai#new_line()")

                # vim.command("call nvim_ai#insert('" + item + "')")
                vim.eval("nvim_ai#insert('" + item + "')")
                tmp_count = tmp_count + 1
    else:
        # vim.command("call nvim_ai#insert('" + script + "')")
        vim.eval("nvim_ai#insert('" + script + "')")

    time.sleep(0.012)

def get_delta_from_res(res):
    try:
        delta = res["choices"][0]["delta"]
        return delta
    except TypeError as e:
        errfile = vim.eval("nvim_ai#errlog_file()")
        traceback.print_exc(file=open(errfile,'a'))
        with open(errfile, 'a') as f:
            output_str = "出错的 res: \n\n" + json.dumps(res) + '\n\n'
            f.write(output_str)
        return {}


def get_valid_json(string):
    res = False
    try:
        res = json.loads(string)
        return res
    except json.JSONDecodeError as e:
        return False


class CustomLLM(LLM):
    logging: bool = False
    output_keys: List[str] = ["output"]
    custom_api: str = ""
    api_key: str = ""
    stream_output: bool = False
    timeout: int = 13
    # 不完整的输出片段
    half_chunk_str: str = ""

    # 支持 openai, apispace, api2d, custom
    llm_type: str = "apispace"

    @property
    def _llm_type(self) -> str:
        return self.llm_type

    def log(self, log_str):
        if self.logging:
            print(log_str)
        else:
            return

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: any = None,
    ) -> str:

        self.log('----------' + self._llm_type + '----------> llm._call()')
        self.log(prompt)

        # custom llm
        if self._llm_type == "custom":
            payload = {
                "prompt": prompt,
                "temperature":0,
                "history": []
            }
            headers = {
                "Content-Type":"application/json"
            }
            try:
                response = requests.request("POST", self.custom_api, data=json.dumps(payload),
                                            headers=headers, timeout=self.timeout)
            except requests.exceptions.Timeout as e:
                vim.command("echom '调用超时'")
                return '{timeout}'

            self.log('<--------custom---------')
            self.log(json.loads(response.text)["response"])
            return json.loads(response.text)["response"]

        # apispace
        elif self._llm_type == "apispace":
            url = "https://eolink.o.apispace.com/ai-chatgpt/create"
            payload = {
                    "system":"你是一个代码生成器，你只会根据我的要求输出代码",
                    "message":["user:" + prompt],
                    "temperature":"0"
                }

            headers = {
                "X-APISpace-Token":self.api_key,
                "Authorization-Type":"apikey",
                "Content-Type":"application/json"
            }
            try:
                response = requests.request("POST", url, data=json.dumps(payload),
                                            headers=headers, timeout=self.timeout)
            except requests.exceptions.Timeout as e:
                vim.command("echom '调用超时'")
                return '{timeout}'

            result = json.loads(response.text)
            if "status" in result and result["status"] == "error":
                vim.command('echom \'' + result["msg"] + '\'')
                return "{error}"
            else:
                self.log('<--------apispace---------')
                self.log(result["result"])
                return result["result"]

        elif self._llm_type == "api2d":
            url = "https://oa.api2d.net/v1/chat/completions"
            gpt_model = vim.eval("g:nvim_ai_model")
            payload = {
                    "model": gpt_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "safe_mode": False
                    }

            headers = {
                'Authorization': "Bearer " + self.api_key,
                'Content-Type': 'application/json'
            }

            # 非流式输出
            if self.stream_output == False:
                try:
                    response = requests.request("POST", url, data=json.dumps(payload),
                                                headers=headers, timeout=self.timeout)
                except requests.exceptions.Timeout as e:
                    vim.command("echom '调用超时'")
                    return '{timeout}'

                result = json.loads(response.text)
                if "object" in result and result["object"] == "error":
                    vim.command('echom "' + result["message"] + '"')
                    return "{error}"
                else:
                    self.log('<--------apispace---------')
                    self.log(result['choices'][0]["message"]["content"])
                    return result['choices'][0]["message"]["content"]

            # 流式输出
            else:
                payload["stream"] = "true"
                try:
                    response = requests.request("POST", url, data=json.dumps(payload),
                                                headers=headers, stream=True, timeout=self.timeout)
                except requests.exceptions.Timeout as e:
                    vim.command("echom '调用超时'")
                    return '{timeout}'

                chunk_chars = ""
                try:
                    vim.command("call nvim_ai#stream_first_rendering()")

                    for chunk in response.iter_content(chunk_size=3000):
                        chunk_chars = self.get_chars_from_chunk(chunk)

                        # TODO: chunk_chars == "" 的情况没有考虑，还不清楚这种情况是否是结束标志
                        # if chunk_chars == "":
                        #     continue

                        if chunk_chars == "[DONE]":
                            vim.command("call nvim_ai#teardown()")
                            vim.command("echom '[DONE]'")
                            return ""
                        elif chunk_chars.endswith("[DONE]"):
                            letters = chunk_chars.replace("[DONE]", "")
                            vim_command_handler(letters)
                            vim_command_handler("[DONE]")
                        else:
                            letters = chunk_chars.replace("'", "''")
                            vim_command_handler(letters)

                except KeyboardInterrupt:
                    print('Interrupted')
                except Exception as e:
                    print(">>:" + str(e))
                    traceback.print_exc(file=open(vim.eval("nvim_ai#errlog_file()"),'a'))

                return ""

        # openai
        elif self._llm_type == "openai":
            api_key = self.api_key
            pass

    def parse_chunk_from_api2d(self, text):
        prefix = "data: "
        output = text
        if text.startswith(prefix):
            output = text[len(prefix):]
        return output.rstrip('\n')

    def get_chars_from_chunk(self, chunk):
        chunk_str = self.parse_chunk_from_api2d(chunk.decode("utf-8"))
        if chunk_str.rstrip() == "[DONE]":
            return "[DONE]"
        try:
            # print('---------------')
            # print(chunk_str)
            result = json.loads(chunk_str)
            delta = result["choices"][0]["delta"]
            if "content" in delta:
                return result["choices"][0]["delta"]["content"]
            else:
                return ""
        except json.JSONDecodeError as e:
            # print("except: jsondecodeerror")
            tmp_data = chunk_str.split("\n")
            curr_letter = ""
            for item in tmp_data:
                if item.strip() == "":
                    continue
                if item.startswith("data:"):
                    line = re.sub(r"^data:", "", item).strip()
                else:
                    line = item.strip()

                if line == "[DONE]":
                    curr_letter = curr_letter + "[DONE]"
                    break

                res = get_valid_json(line)
                if res == False:
                    # print('出现了截断的情况')
                    # 出现了被截断的情况
                    if re.compile(r'^{.id.:').search(line) == None:
                        # 头部被截断，则补上头部
                        # print('头部被截断，补充上头部')
                        line = self.half_chunk_str + line
                        self.half_chunk_str = ""
                        # print(line)
                        res = get_valid_json(line)
                        if res == False:
                            continue

                    else:
                        # print("尾部被截断，把片段保存为头部")
                        # print(line)
                        # 尾部截断，则保存为头部
                        self.half_chunk_str = line
                        continue

                # 正常的完整JSON
                delta = get_delta_from_res(res)
                if "content" in delta:
                    curr_letter = curr_letter + delta["content"]

            return curr_letter


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": 10}

