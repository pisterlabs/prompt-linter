from .anpl import ANPL
from copy import deepcopy
import openai
from openai.error import RateLimitError
from typing import Optional
import time
import re
 
def msg(role: str, content: str): return {"role": role, "content": content}

def is_valid_python_syntax(string):
    try:
        compile(string, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

pattern = re.compile(r"```python(.+?)```", flags=re.DOTALL)
def extract_code(text: str):
    # print(text)
    codes =[match.strip() for match in re.findall(pattern, text)]
    if len(codes) > 0:
        code = "\n".join(codes)
        if is_valid_python_syntax(code):
            return code
    return None

def collect(res: dict): return [extract_code(choice["message"]["content"]) for choice in res["choices"]]

def raw_query(msgs: list[dict], retries=3, **config) -> list[Optional[str]]:
    for i in range(retries):
        try:
            res = openai.ChatCompletion.create(messages=msgs, **config)
            return res
        except RateLimitError:
            if i == retries - 1:
                raise RateLimitError
            print("start sleep")
            time.sleep(20 * (2 ** i))
            print("sleep end")

def query(msgs: list[dict], retries=3, **config) -> list[Optional[str]]:
    return collect(raw_query(msgs, retries, **config))

def hole_coder(source_code, hole):
    system_prompt = 'As a pythonGPT, your task is to complete the unimplemented functions in the given python code, which are referred to as "holes" and are labeled as _hole0, _hole1, _hole2, and so on. Your implementation should align with the code and documentation using Python.'
    user_prompt = """```python
{code}
```
The function needs to be given a new name. Markdown format should be used to return it.
```python
{hole}
```"""
    return [msg("system", system_prompt), msg("user", user_prompt.format(code=source_code, hole=hole))]

def hole_resyner(source_code, hole):
    system_prompt = 'As a pythonGPT, your task is to complete the unimplemented functions in the given python code, which are referred to as "holes". Your implementation should align with the code and documentation using Python.'
    user_prompt = """```python
{code}
```
Markdown format should be used to return it.
```python
{hole}
```"""
    return [msg("system", system_prompt), msg("user", user_prompt.format(code=source_code, hole=hole))]

def fun_synthesis(anpl: ANPL, name: str, temp:float = 0) -> Optional[str]:
    actor = hole_coder if name.startswith("_hole") else hole_resyner
    msgs = actor(anpl.to_python(), f"# {anpl.funs[name].prompt}\ndef {name}")
    res = query(msgs, model="gpt-3.5-turbo", max_tokens=1024, temperature=temp)[0]
    return res

def batch_fun_synthesis(anpl: ANPL, name: str, n:int, temp:float) -> list[Optional[str]]:
    anpl_for_prompt = deepcopy(anpl)
    actor = hole_coder if name.startswith("_hole") else hole_resyner
    anpl_for_prompt.funs[name].reset_to_hole()
    msgs = actor(anpl_for_prompt.to_python(), f"# {anpl.funs[name].prompt}\ndef {name}")
    reses = query(msgs, model="gpt-3.5-turbo", max_tokens=1024, n=n, temperature=temp)
    assert len(reses) == n, "OpenAI doesn't return enough responce"
    return reses
