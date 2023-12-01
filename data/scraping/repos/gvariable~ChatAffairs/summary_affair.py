import openai
from dotenv import dotenv_values
import asyncio
from typing import List, Dict
import json
import json_stream
from pathlib import Path
from tqdm.asyncio import tqdm
import backoff

CONFIG = dotenv_values(".env")
openai.api_key = CONFIG["OPENAI_API_KEY"]


def get_meta(fn):
    """
    Open the file specified by `fn` and yield each metadata object from the JSON stream.
    """
    with open(fn) as f:
        for meta in json_stream.load(f):
            meta = json_stream.to_standard_types(meta)
            yield meta


@backoff.on_exception(backoff.expo, Exception, max_time=240)
async def get_completion(prompt, model="gpt-3.5-turbo-16k"):
    messages = [{"role": "user", "content": prompt}]

    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=messages,
        temperature=0.4,  # the lower, the better?
    )

    return response.choices[0].message["content"]


async def get_completions(metas: List[Dict]) -> List[str]:
    # TODO(gpl): change metas to generator
    basic_prompt = """
你的任务是总结湖北省政务文件的内容，构造两个问答对，并使用json格式输出问答对。
构造的回答应该尽量详细且至少包含办理政务的地点、时间、所需材料等，地点信息尽可能详细，精确到楼栋，请保证输出能清晰解释相关政务事项。
提取政务标题内容为<title>，构造的问题应该类似于：
- 我想了解一下<title>
- <title>办理的整个流程是什么样的
- 告诉我关于<title>的基本信息    
输出示例如下：
{"input": 问题, "output": 回答}
湖北省政务文件是json格式，内容如下：
"""
    prompts = [basic_prompt + json.dumps(meta, ensure_ascii=False) for meta in metas]

    completions = []
    for coro in tqdm.as_completed(
        [get_completion(prompt) for prompt in prompts], leave=False
    ):
        completions.append(await coro)

    return completions


async def main():
    fn = Path("./dataset/data.json")
    workers = 200

    with open("tmp.jsonl", "w") as f:
        samples = []
        for idx, meta in tqdm(enumerate(get_meta(fn))):
            samples.append(meta)
            if (idx + 1) % workers == 0:
                completions = await get_completions(samples)
                f.write("\n".join(completions))
                samples = []

        if samples:
            completions = await get_completions(samples)
            f.write("\n".join(completions))


if __name__ == "__main__":
    asyncio.run(main())
