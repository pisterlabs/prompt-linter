r"""Translate text to English.

https://beta.openai.com/playground/p/default-translate
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nFrench:"
restart_sequence = "\n\nEnglish: "

prompt="English: I do not speak French.\nFrench: Je ne parle pas français.\n\nEnglish: See you later!\nFrench: À tout à l'heure!\n\nEnglish: Where is a good restaurant?\nFrench: Où est un bon restaurant?\n\nEnglish: What rooms do you have available?\nFrench: Quelles chambres avez-vous de disponible?\n\nEnglish: this is a test\nFrench: c'est un test\n\nEnglish: this is anotther test\nFrench: "

prompt="English: I do not speak French.\nFrench: Je ne parle pas français.\n\nEnglish: See you later!\nFrench: "

query = "你叫什么名字？"

prompt=f'''\
Chinese: 我不会英文。
English: I don't speak English.

Chinese: 我
English: me

Chinese: {query}
English: \
'''

engine="davinci-instruct-beta-v3"
engine="davinci"

response = openai.Completion.create(
  prompt=prompt,
  engine=engine,
  temperature=0.2,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  # stop=["###"]  # need this for engine="davinci"？
  stop=["\n\n"]  # need this for engine="davinci"？
)
response.choices[0].text

"""
from typing import List

import os
import sys
import re
from textwrap import dedent
import dotenv
import openai

import logzero
from logzero import logger

_ = os.environ.get("LOGLEVEL", 20)
if isinstance(_, str):
    dict_ = zip(["debug", "info", "warning", "error", "critical"], range(10, 60, 10))
    _ = dict(dict_).get(_.lower(), _)

try:
    _ = int(_)
except ValueError:
    _ = 20
logzero.loglevel(_)

_ = "OPENAI_API_KEY"
OPENAI_API_KEY = os.environ.get(_) or dotenv.dotenv_values().get(_)
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set, you can set in .env or system environ")
else:
    _ = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}"
    logger.debug("OPENAI_API_KEY: %s", _)
    openai.api_key = OPENAI_API_KEY


# fmt: off
def zh2en(
        query: str,
        # engine: str = "davinci-instruct-beta-v3"
        engine: str = "davinci",
        temperature: float = 0.11,
        max_tokens: int = 150,
        top_p: int = 1,
        frequency_penalty: float = 0,  # -2..2, positive penalty
        presence_penalty: float = 0,
        stop: List[str] = ["###", "\n\n"],
) -> str:
    # fmt: on
    r"""Translate Chinese to English via gpt2.

    Args
        query: text to translate, newlines (\n) will
            be removed; occurrences of ## in query
            cause problems. Hence, ## is replacceed with #-.
        others: refer to openai.Completion.create docs,
            engine "davinci-instruct-beta-v3" is better but collects data
            higher temperature, more dynamic (more variants)

    Returns
        translated English, may occasionally return blank,
        a simple retry or retry with different temperature
        often solves the problem.
    """
    # query = str(query).replace("##", "")

    # replace ##, ###,..., with #- #--, ...
    # (credit to Konge for this regex trick)
    # query = re.sub(r"(?<=(#))\1", "-", str(query))

    # credit to all netpals
    query = re.sub(r"(?<=#)#", "-", str(query))

    # remove all \n
    # query = query.repalce("\n", " ", query)
    query = re.sub("\n+", " ", query)

    prompt = dedent(f"""\
        Chinese: 我
        English: Me
        ###
        Chinese: {query}
        English:\
    """).strip() + " "

    # prompt = f"Chinese: 我\nEnglish: Me\n###\nChinese: {query}\nEnglish: "  # works with stop="###"

    # prompt = f"Chinese: 我\nEnglish: Me\n\nChinese: {query}\nEnglish: "

    logger.debug("prompt: [%s]", prompt)

    try:
        resp = openai.Completion.create(
            prompt=prompt,
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            # stop=["##"],
            # stop="\n\n",
        )
        logger.debug("resp: %s", resp)
    except Exception as exc:
        logger.error(exc)
        return str(exc)

    try:
        res = resp.choices[0].text
    except Exception as exc:
        logger.error(exc)
        res = str(exc)

    return res.strip()


def main():  # noqa
    if sys.argv.__len__() > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "世界卫生组织26日将最新报告的新冠病毒变异毒株B.1.1.529列为“需要关注”的变异毒株，并以希腊字母“奥密克戎”（Ο）命名。目前，这一变异毒株已在三大洲被发现，多国采取入境限制措施。"

    logger.info("%s -> %s", query, zh2en(query))


if __name__ == "__main__":
    main()
