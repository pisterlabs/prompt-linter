from google.cloud import vision
from google.oauth2 import service_account
import guidance
import os
import streamlit as st
import pandas as pd
import io
import codecs
import json


def csv_string_to_df(csv_string):
    return pd.read_csv(io.StringIO(csv_string))


def add_bom(csv_string: str):
    return (codecs.BOM_UTF8 + csv_string.encode("utf-8")).decode("utf-8")


guidance.llm = guidance.llms.OpenAI("gpt-4-0613", api_key=os.environ["OPENAI_API_KEY"])

credentials_json = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
credentials_info = json.loads(credentials_json)
credentials = service_account.Credentials.from_service_account_info(credentials_info)

client = vision.ImageAnnotatorClient(credentials=credentials)


def detect_document(content):
    """Detects document features in an image."""

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    blocks = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                # paragraph.wordsについて、同じy座標のものをまとめる
                words_grouped_by_y = {}
                for word in paragraph.words:
                    y = word.bounding_box.vertices[0].y
                    if y in words_grouped_by_y:
                        words_grouped_by_y[y].append(word)
                    else:
                        words_grouped_by_y[y] = [word]

                for words in words_grouped_by_y.values():
                    blocks.append(
                        {
                            "x": words[0].bounding_box.vertices[0].x,
                            "y": words[0].bounding_box.vertices[0].y,
                            "text": "".join(
                                [
                                    "".join([symbol.text for symbol in word.symbols])
                                    for word in words
                                ]
                            ),
                        }
                    )

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return blocks


def retrieve_table(blocks):
    prompt = guidance(
        """
{{#system}}あなたは便利なアシスタントです。ユーザーの手伝いをしてあげてください。{{/system}}

{{#user}}
司令: 領収書をOCRにかけた結果を渡します。
読み取られた文字列を、「x座標,y座標: ラベル」の形式で書かれています。
ここから、領収書の内容を抜き出してCSVにしてください。CSVにする際は、数字に含まれるカンマでセルが分割されないように、数字はダブルクォートで囲ってください。

また、抜き出したCSVの前後に[csv]と[/csv]をマーカーとしてつけてください。

表の内容には、項目名と金額が含まれます。

OCR結果:
{{#each blocks}}{{this.x}},{{this.y}}: {{this.text}}\n{{/each~}}
{{/user}}

{{#assistant}}{{gen "result"}}{{~/assistant}}
"""
    )
    result = prompt(blocks=blocks)

    text = result["result"]

    return text.split("[csv]")[1].split("[/csv]")[0].strip()


file = st.file_uploader("画像ファイルを選択")

if file is not None:
    st.image(file, caption="選択された画像", use_column_width=True)

    loading_text = st.text("読み取り中...")

    ocr_blocks = detect_document(file.getvalue())

    with st.expander("生データ"):
        st.write(
            [f'{block["x"]},{block["y"]}: {block["text"]}' for block in ocr_blocks]
        )

    csv = retrieve_table(ocr_blocks)

    loading_text.text("✅ DONE")

    st.write(
        """## 読み取り結果

```csv
{}
```
""".format(
            csv
        )
    )

    st.download_button(
        "CSVをダウンロード", add_bom(csv), "table-extract.csv", "text/csv", key="download-csv"
    )

    st.text("プレビュー")

    st.dataframe(csv_string_to_df(csv))
