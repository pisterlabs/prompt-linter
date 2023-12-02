import time
import json
import openai
import threading
from textwrap import dedent
from src.util import chat_completion_request


def generate_summary(placeholder, title, abst, list_of_things_to_ask: list):
    _question_list = "\n".join([
                   f"({i}){text}"
                   for i, text in enumerate(list_of_things_to_ask)
               ])
    prompt = """
    以下の論文について何がすごいのか、次の項目を日本語で出力してください。

    {questions}


    タイトル: {title}
    アブストラクト: {abst}
    日本語で出力してください。
    """.format(title=title, abst=abst, questions=_question_list)

    _funcs = {
        f"variable_{i}": _question
        for i, _question in enumerate(list_of_things_to_ask)
    }
    functions = [
        {
            "name": "format_output",
            "description": "アブストラクトのサマリー",
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": "string",
                        "description": _question,
                    }
                    for name, _question in _funcs.items()
                },
                "required": list(_funcs.keys()),
            },
        }
    ]

    summary_prompt = """
    以下のアブストラクトの全文を一文ずつ分かりやすく日本語訳してください．文の順序は変えないでください．

    アブストラクト: {abst}

    日本語で出力してください。
    """.format(title=title, abst=abst, questions=_question_list)
    functions2 = [
        {
            "name": "format_output",
            "description": "アブストラクトの日本語訳",
            "parameters": {
                "type": "object",
                "properties": {
                    "translation_to_japanese": {
                        "type": "string",
                        "description": "アブストラクトの全文を一文ずつ日本語訳したもの",
                    },
                },
                "required": ["translation_to_japanese"],
            },
        }
    ]

    placeholder.markdown("ChatGPTが考え中です...😕", unsafe_allow_html=True)

    #res = chat_completion_request(messages=[{"role": "user", "content": prompt}], functions=functions)
    m = [{"role": "user", "content": prompt}]
    m2 = [{"role": "user", "content": summary_prompt}]
    result1 = []
    result2 = []
    thread = threading.Thread(target=chat_completion_request, args=(m, functions, result1))
    thread2 = threading.Thread(target=chat_completion_request, args=(m2, functions2, result2))
    thread.start()
    thread2.start()
    i = 0
    faces = ["😕", "😆", "😴", "😊", "😱", "😎", "😏"]
    while thread.is_alive() or thread2.is_alive():
        i += 1
        face = faces[i % len(faces)]
        placeholder.markdown(f"ChatGPTが考え中です...{face}", unsafe_allow_html=True)
        time.sleep(0.5)
    thread.join()
    thread2.join()

    if len(result1) == 0:
        placeholder.markdown("ChatGPTの結果取得に失敗しました...😢", unsafe_allow_html=True)
        return

    if len(result2) == 0:
        placeholder.markdown("ChatGPTの結果取得に失敗しました...😢", unsafe_allow_html=True)
        return

    res = result1[0]
    res2 = result2[0]
    func_result = res.json()["choices"][0]["message"]["function_call"]["arguments"]
    func_result2 = res2.json()["choices"][0]["message"]["function_call"]["arguments"]
    output = json.loads(func_result)
    output2 = json.loads(func_result2)

    translation_to_japanese = output2["translation_to_japanese"]
    output_elements = dedent("".join(
        [
            f"""<li><b>アブストラクトの日本語訳</b></li><li style="list-style:none;">{translation_to_japanese}</li>"""
         ] + [
        dedent(f"""<li><b>{question}</b></li><li style="list-style:none;">{output[name]}</li>""")
        for name, question in _funcs.items()
    ]))
    gen_text = dedent(f"""以下の項目についてChatGPTが回答します。
    <ul>{output_elements}</ul>"""
    )
    print(gen_text)
    render_text = f"""<div style="border: 1px rgb(128, 132, 149) solid; padding: 20px;">{gen_text}</div>"""
    placeholder.markdown(dedent(render_text), unsafe_allow_html=True)
    return gen_text
