from langchain import PromptTemplate, FewShotPromptTemplate, OpenAI

from fastapi import FastAPI

app = FastAPI()  # FastAPIのインスタンスを作成


# 教師データ
examples = [
    {
        "name": "Notion",
        "feature": "- オールインワークスペース: ノート、タスク、データベース、カレンダーなどを1つのプラットフォームで提供。\n- カスタマイズ可能: ページの自由なデザインや、必要なブロックの追加が可能。\n- 協力作業: チームメンバーとの共同編集やコメント、タスクの割り当て機能を持つ。",
        "examples": "- 知識ベースの作成: チームのナレッジやガイドラインを中央で管理。\n- タスク管理: プロジェクトのタスクや進捗をトラック。\n- 個人のノート取り: アイディアやリサーチのメモを整理。"
    },
]

# 教師データのフォーマット
tool_formatter_template = """
## 名前
{name}

## 特徴
{feature}

## 使用例
{examples}
"""

# PromptTemplateのインスタンスを作成
tool_prompt_template = PromptTemplate(
    template=tool_formatter_template,
    input_variables=["name", "feature", "examples"]
)

# FewShotPromptTemplateのインスタンスを作成
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=tool_prompt_template,
    prefix="ツールの詳細を教えてください。",
    suffix="ツール名: {input}",
    input_variables=["input"],
    example_separator="\n\n",
)

# テンプレートをフォーマット
prompt_text = few_shot_prompt.format(input="ChatGPT")
print(prompt_text)

# OpenAIのモデルのインスタンスを作成
llm = OpenAI(model_name="text-davinci-003", openai_api_key="xxxx")

# モデルにプロンプトを送信し、結果を表示
print(llm(prompt_text))
