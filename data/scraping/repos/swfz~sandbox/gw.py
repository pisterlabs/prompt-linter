import openai
import sys

subcommand = sys.argv[1]
markdown_file = sys.argv[2]

def writing(markdown_file):
    with open(markdown_file, 'r', encoding="utf-8") as f:
        mddata = f.read()

        system_content = """
あなたは技術記事の執筆を行うソフトウェアエンジニアです。
与えられたテキストをもとに、章立て、技術記事の執筆をしてください。

テキストの例:

# タイトル
- リストの項目1つ目
- リストの項目2つ目
- リストの項目3つ目
"""
        res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": mddata},
                ]
            )

        print(res.choices[0]["message"]["content"].strip())

def convert(markdown_file):
    with open(markdown_file, 'r', encoding="utf-8") as f:
        mddata = f.read()

        system_content = """
あなたは技術記事の執筆を行うソフトウェアエンジニアです。
与えられたテキストを文語体で書いてください
"""
        res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": mddata},
                ]
            )

        print(res.choices[0]["message"]["content"].strip())



def summarize(markdown_file):
    with open(markdown_file, 'r', encoding="utf-8") as f:
        mddata = f.read()

        system_content = """
あなたは技術記事の執筆を行うソフトウェアエンジニアです。
これからあなたに実際にコードを書いて動作せせるための過程やタイトルの結論に到るまでの様々な試行錯誤の履歴のメモを渡します
与えられたメモから、話を要約し、技術記事の執筆をしてください。
また、文語体で書いてください
"""
        res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": mddata},
                ]
            )

        print(res.choices[0]["message"]["content"].strip())


def tags(markdown_file):
    with open(markdown_file, 'r', encoding="utf-8") as f:
        mddata = f.read()

        system_content = """
あなたはアノテーターです。
与えられたテキストから、話を要約し、タグを付けてください
"""
        res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": mddata},
                ]
            )

        print(res.choices[0]["message"]["content"].strip())

def calibration(markdown_file):
    with open(markdown_file, 'r', encoding="utf-8") as f:
        mddata = f.read()

        system_content = """
あなたは技術雑誌の編集者です。
与えられた記事のテキストを校正し、修正した方が良い箇所を指摘してください
また、指摘箇所と指摘後の行を並べてDiffファイル形式で出力してください
"""
        res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": mddata},
                ]
            )

        print(res.choices[0]["message"]["content"].strip())

print(subcommand)
print('----------')

if subcommand == "writing":
    writing(markdown_file)

if subcommand == "convert":
    convert(markdown_file)

if subcommand == "summarize":
    summarize(markdown_file)

if subcommand == "tags":
    tags(markdown_file)

if subcommand == "calibration":
    calibration(markdown_file)
