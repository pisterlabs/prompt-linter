import json
import os
import re
import requests
import time

from bs4 import BeautifulSoup
import deepl
import openai

from func_package.deepl_func import translate_2en, translate_2ja
from func_package.extract_text import (
    extract_latter_half,
    extract_first_thirds,
    extract_middle_thirds,
    extract_last_thirds,
)
from func_package.openai_func import (
    paraphrase_en,
    format_html,
    write_continue,
    paraphrase_title,
)
from func_package.wp_api_func import update_with_html

### 認証まわり
# DeepLの認証とtranslatorオブジェクトの生成
deepl_api_key = os.environ["DEEPL_API_KEY"]
translator = deepl.Translator(deepl_api_key)

# OPEN AIオブジェクトに認証キー情報をもたせる
openai.api_key = os.environ["OPENAI_API_KEY"]

# WordPressのログイン情報を読み込み
with open("wp_login_info.json", "r") as f:
    wp_login_info = json.load(f)
username = wp_login_info["username"]
password = wp_login_info["password"]
wp_root_url = wp_login_info["wp_root_url"]


### パラフレーズされたテキストの作成
# 投稿IDをループで取得し、投稿IDのリストを作成
with open("./wp-post-ids.txt") as f:
    post_ids = f.read()
post_ids_list = post_ids.split(" ")

# 途中から開始（任意のリスト番号に変更）
# post_ids_list = post_ids_list[10:]

# パラフレーズ〜WordPressへの更新を全記事に対してパラフレーズを実行
for post_id in post_ids_list:
    # DeepLで日本語から英語に翻訳
    with open(f"original_ja_contents/{post_id}") as f:
        content_text = f.read()
    translated_en = translate_2en(translator, content_text)

    # 3分割して、OPEN AIで英語文章をパラフレーズ
    translated_en_1 = extract_first_thirds(translated_en)
    translated_en_2 = extract_middle_thirds(translated_en)
    translated_en_3 = extract_last_thirds(translated_en)

    paraphrased_text_1 = paraphrase_en(openai, translated_en_1)
    time.sleep(5)
    paraphrased_text_2 = paraphrase_en(openai, translated_en_2)
    time.sleep(5)
    paraphrased_text_3 = paraphrase_en(openai, translated_en_3)
    time.sleep(4)

    # OPEN AIで続きの文章を生成
    last_part = extract_latter_half(paraphrased_text_3)
    continue_text = write_continue(openai, last_part)

    # DeepLでそれぞれを英語から日本語へ再翻訳: 空白文が渡るとValueError
    retranslated_ja_1 = translate_2ja(translator, paraphrased_text_1)
    time.sleep(3)
    retranslated_ja_2 = translate_2ja(translator, paraphrased_text_2)
    time.sleep(3)
    retranslated_ja_3 = translate_2ja(translator, paraphrased_text_3)
    time.sleep(3)
    if len(continue_text) > 20:
        retranslated_ja_4 = translate_2ja(translator, continue_text)

    # それぞれを3分割し、OpenAIでHTML形式のテキストへフォーマット
    response_html_1 = format_html(openai, extract_first_thirds(retranslated_ja_1))
    time.sleep(4)
    response_html_2 = format_html(openai, extract_middle_thirds(retranslated_ja_1))
    time.sleep(4)
    response_html_3 = format_html(openai, extract_last_thirds(retranslated_ja_1))
    time.sleep(4)

    response_html_4 = format_html(openai, extract_first_thirds(retranslated_ja_2))
    time.sleep(4)
    response_html_5 = format_html(openai, extract_middle_thirds(retranslated_ja_2))
    time.sleep(4)
    response_html_6 = format_html(openai, extract_last_thirds(retranslated_ja_2))
    time.sleep(4)

    response_html_7 = format_html(openai, extract_first_thirds(retranslated_ja_3))
    time.sleep(4)
    response_html_8 = format_html(openai, extract_middle_thirds(retranslated_ja_3))
    time.sleep(4)
    response_html_9 = format_html(openai, extract_last_thirds(retranslated_ja_3))

    if retranslated_ja_4:
        time.sleep(4)
        response_html_10 = format_html(openai, extract_first_thirds(retranslated_ja_4))
        time.sleep(4)
        response_html_11 = format_html(openai, extract_middle_thirds(retranslated_ja_4))
        time.sleep(4)
        response_html_12 = format_html(openai, extract_last_thirds(retranslated_ja_4))

    # 生成されたHTMLテキストをすべて連結
    if retranslated_ja_4:
        response_html_whole = (
            response_html_1
            + "<br>"
            + response_html_2
            + "<br>"
            + response_html_3
            + "<br>"
            + response_html_4
            + "<br>"
            + response_html_5
            + "<br>"
            + response_html_6
            + "<br>"
            + response_html_7
            + "<br>"
            + response_html_8
            + "<br>"
            + response_html_9
            + "<br>"
            + response_html_10
            + "<br>"
            + response_html_11
            + "<br>"
            + response_html_12
        )
    else:
        response_html_whole = (
            response_html_1
            + "<br>"
            + response_html_2
            + "<br>"
            + response_html_3
            + "<br>"
            + response_html_4
            + "<br>"
            + response_html_5
            + "<br>"
            + response_html_6
            + "<br>"
            + response_html_7
            + "<br>"
            + response_html_8
            + "<br>"
            + response_html_9
        )

    # OpenAIでオリジナル日本語タイトルからタイトルを生成
    with open(f"./original_ja_title/{post_id}") as f:
        title_original = f.read()
    title_created = paraphrase_title(openai, title_original)

    ### WordPressへのUpdateを実行
    # エンドポイントを定義
    api_update_url = f"{wp_root_url}/wp-json/wp/v2/posts/{post_id}"

    ## Updateの実行: 公開状態に
    json_html_body = {
        "title": title_created,
        "content": response_html_whole,
        "status": "publish",
    }
    returned_post_obj = update_with_html(
        requests, api_update_url, username, password, json_html_body
    )

    # 実行結果を出力
    print(
        f"Success! Post updated.\nPost ID: {returned_post_obj['id']}; URL: {returned_post_obj['link']}\nTitle: {returned_post_obj['title']['rendered']}\n------"
    )
