import json
import os

import requests
from PIL import Image
import deepl
import openai

from func_package.deepl_func import translate_2en
from func_package.pixabay_image import (
    check_result_is_0,
    extract_image_url,
    search_pixabay,
)
from func_package.openai_func import extract_keyword, synonym_keyword
from func_package.wp_api_func import update_api_access

## 認証まわり
# Pixabay
pixabay_url = "https://pixabay.com/api/"
pixabay_api_key = os.environ["PIXABAY_API_KEY"]

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


## 日本語タイトルから英語キーワード抽出: DeepL&OpenAI
with open("./wp-post-ids.txt") as f:
    post_ids = f.read()
post_ids_list = post_ids.split(" ")

# 途中から開始（任意のリスト番号に変更）
# post_ids_list = post_ids_list[10:]

# 投稿IDをループで取得し実行
for post_id in post_ids_list:
    with open(f"./original_ja_title/{post_id}") as f:
        original_title = f.read()

    # DeepLで翻訳を実行
    en_title = translate_2en(translator, original_title)

    # 代表キーワードを抽出: OpenAI
    keyword = extract_keyword(openai, en_title)

    ## 画像の検索・取得
    # Pixabay: キーワード検索
    # 代表キーワード1語で検索
    response = search_pixabay(requests, pixabay_api_key, keyword)

    keyword_result_is_0 = check_result_is_0(response)
    if keyword_result_is_0:
        # Open AIで類義語リストの生成
        response_words = synonym_keyword(openai, keyword)
        synonym_list = response_words.lstrip("[,'").rstrip("']").split("', '")
        # 類義語リストでキーワード検索
        for synonym in synonym_list:
            response = search_pixabay(requests, pixabay_api_key, synonym)
            # 検索結果の判定
            synonym_result_is_0 = check_result_is_0(response)
            if synonym_result_is_0:
                continue
            else:
                image_url = extract_image_url(response)
                break
        else:
            print("Cannot detect images in any synonyms.")
    else:
        image_url = extract_image_url(response)

    ## 画像の保存・変換
    # 指定画像をバイナリ形式でローカルにDL
    response = requests.get(image_url)

    if response.status_code == 200:
        with open(f"./pixabay_images_binary/{post_id}", "wb") as f:
            f.write(response.content)

    ## 保存したバイナリファイルをPNGに変換
    image = Image.open(f"./pixabay_images_binary/{post_id}")
    # RGBAイメージをRGBイメージに変換
    rgb_image = image.convert("RGB")
    # pngで保存
    rgb_image.save(f"./pixabay_images_png/{post_id}.png", format="PNG")

    ### WordPress投稿へ画像をアップロード
    ## メディアライブラリに画像をアップロード
    # png画像ファイルの読み込み
    with open(f"./pixabay_images_png/{post_id}.png", "rb") as f:
        img_data = f.read()

    headers = {
        "Content-Disposition": f"attachment; filename={post_id}.png",
        "Content-Type": "image/png",
    }

    # メディアライブラリへのアップロード実行
    url_media = f"{wp_root_url}/wp-json/wp/v2/media"
    media_response = requests.post(
        url_media, auth=(username, password), headers=headers, data=img_data
    )

    ## アップロードしたファイルと投稿サムネイルを紐付け
    update_url = f"{wp_root_url}/wp-json/wp/v2/posts/{post_id}"
    media_dict = media_response.json()
    post_data = {"featured_media": media_dict["id"]}

    # 紐付けの実行（update）
    post_dict = update_api_access(requests, update_url, username, password, post_data)

    # 実行結果を出力
    print(
        f"Success! Thumbnail updated.\nPost ID: {post_dict['id']}; URL: {post_dict['link']}\nTitle: {post_dict['title']['rendered']}\n------"
    )
