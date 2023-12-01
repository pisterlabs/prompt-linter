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
    你的任务是从湖北省政务文件中提取关键信息，整理出问题及对应的答案，并使用jsonl的输出格式，你可以按照以下步骤进行：
    1. 编写问题和回答对：从湖北省政务文件中提取关键信息，编写8-12对问题和对应的详细回答，请额外关注办理政务的地点、时间、材料、人员等信息。问题中应携带政务标题，回答应该尽量详细且能够清晰解释相关政务事项。
    2. 输出为jsonl格式： 将每个问题和回答对整理成一个json对象，并按照jsonl格式输出到文件中。使用"input"作为键来表示问题，使用"output"作为键来表示回答，不同问题和回答之间使用\n分隔。例如：
    {"input": 问题1, “output”: 回答1}\n{"input": 问题2, “output”: 回答2}
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
    fn = Path("data.json")
    workers = 15

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
