import openai
import tweepy
import requests
import random
import re
from io import BytesIO
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import sys
import importlib

# ログの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# StreamHandlerの設定
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)

# ハンドラをロガーに追加
logger.addHandler(handler)

def get_all_entries(feed_url):
    """
    指定されたRSSフィードのすべての記事を取得する関数
    """
    # 環境変数からblog_typeを取得
    blog_type = os.environ.get("BLOG_TYPE")

    try:
        # ブログタイプに対応するモジュールを動的にインポート
        processor_module = importlib.import_module(f"entry_processor_{blog_type}")
        # モジュールから処理関数を取得
        process_blog = getattr(processor_module, f"get_all_entries")
    except (ImportError, AttributeError):
        # モジュールが存在しない場合、またはモジュール内に処理関数が存在しない場合は例外をスロー
        raise Exception(f"Blog type {blog_type} is not supported.")

    # 関数を実行してエントリを取得
    entries = process_blog(feed_url)

    logging.info("Finished getting all entries from feed")
    return entries

def get_system_role_for_extracting_image_url():
    # 環境変数からblog_typeを取得
    blog_type = os.environ.get("BLOG_TYPE")

    try:
        # ブログタイプに対応するモジュールを動的にインポート
        processor_module = importlib.import_module(f"entry_processor_{blog_type}")
        # モジュールから処理関数を取得
        process_blog = getattr(processor_module, f"get_system_role_for_extracting_image_url")
    except (ImportError, AttributeError):
        # モジュールが存在しない場合、またはモジュール内に処理関数が存在しない場合は例外をスロー
        raise Exception(f"Blog type {blog_type} is not supported.")
    
    return process_blog()

def get_random_entry(entries):
    """
    ランダムな記事を取得する関数
    """
    if entries:
        entry = random.choice(entries)
        logger.info(f"Random entry selected: {entry['link']}")
        return entry
    else:
        logger.warning("No entries provided to select a random entry.")
        return None

def get_specific_entry(entries, article_url):
    """
    取得した記事の中から、指定されたURLの記事を取得する関数
    """
    for entry in entries:
        if entry['link'] == article_url:
            logger.info(f"Specific entry found: {article_url}")
            return entry
    logger.warning(f"Specific entry not found for URL: {article_url}")
    return None

def split_text_approximately(text, max_tokens):
    chunk_size = int(max_tokens * 0.95)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    text_chunks = text_splitter.split_text(text)
    selected_chunk = random.choice(text_chunks)
    logger.info(f"Text split into {len(text_chunks)} chunks, one random chunk selected for use.")
    return selected_chunk

def generate_tips(article_content, aricle_url, hash_tags):
    """
    OpenAIを使用してTIPSを生成する関数
    """
    logger.info("Generating tips using OpenAI")

    # OpenAI APIの設定
    openai.api_type = os.environ.get("OPENAI_API_TYPE")
    openai.api_base = os.environ.get("OPENAI_API_BASE")
    openai.api_version = os.environ.get("OPENAI_API_VERSION")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Few-shots Learningによって、回答例を学習させるためのプロンプトを設定します。
    query_prompt_few_shots = [
        {"role" : "user", "content" : FEW_SHOTS_USER_01 },
        {"role" : "assistant", "content" : FEW_SHOTS_ASSISTANT_01 },
        {"role" : "user", "content" : FEW_SHOTS_USER_02 },
        {"role" : "assistant", "content" : FEW_SHOTS_ASSISTANT_02 },
    ]

    aricle_url_without_scheme = re.sub(r"^https?://", "", aricle_url)

    question = f"""
# 本文
{article_content}

# 記事のURL
{aricle_url_without_scheme}

# ハッシュタグ
{hash_tags}
"""
    messages = []

    messages.append({"role": "system", "content": SYSTEM_ROLE})

    for shot in query_prompt_few_shots:
        messages.append(shot)

    messages.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        deployment_id=os.environ.get("DEPLOYMENT_ID"),
        messages=messages
    )

    tips = response.choices[0].message["content"]

    # 記事の本文から最後の行にある画像のURLを取得する
    image_url = tips.strip().split('\n')[-1]

    # 画像のURLを取り除く
    tips_without_image_url = tips.replace(image_url, '')

    logger.info("Finished generating tips")
    return (image_url, tips_without_image_url.strip())

def create_tweet_with_image(api, client, text, image_url):
    """
    画像付きのツイートを作成する関数
    """
    logger.info("Creating tweet with image")

    if image_url != "no image":
        logger.info(f"Image URL provided: {image_url}")
        try:
            image_data = requests.get(image_url).content
            image_file = BytesIO(image_data)
            media = api.media_upload("image.jpg", file=image_file)
            client.create_tweet(text=text, media_ids=[media.media_id])
        except Exception as e:
            logger.error(f"Failed to download or upload image: {e}")
            logger.info("Creating text-only tweet due to image error.")
            client.create_tweet(text=text)
    else:
        logger.info("No image URL provided. Creating text-only tweet.")
        client.create_tweet(text=text)

if __name__ == "__main__":
    # OpenAIに与えるsystemのroleを設定します。
    additional_system_role = os.environ.get("ADDITIONAL_SYSTEM_ROLE", "")
    system_role_for_extracting_image_url = get_system_role_for_extracting_image_url()
    SYSTEM_ROLE = f"""
あなたは、与えられたブログの記事から、なにか画像を一つ引用して、そのブログにある役に立つTIPSを一つ作成するAIアシスタントです。
プロンプトには、「記事の本文」「記事のURL」「ハッシュタグ」が与えられます。それをもとに、記事内にある画像を引用した役に立つTIPSを一つ作成してください。これを「TIPS」と呼びます。

次に、「TIPS」の後に「詳しくはこのブログを見てね!」という文言とともに、記事のURLを挿入してください。これを「記事のURLへのリンク」と呼びます。

次に、「記事のURLへのリンク」の後に「# ハッシュタグ」で指定されたハッシュタグを挿入してください。これを「ハッシュタグ」と呼びます。

ここまでで生成したテキスト(TIPS + 記事のURLへのリンク + ハッシュタグ)は、X(旧Twitter)の制限である140字を超えないようにして下さい。

そして最後に一旦改行して、最後の行には引用した画像のURLを挿入してください。挿入するURLには余計な文字は入れないでください。{system_role_for_extracting_image_url}

もし引用するための適当な画像が見つからない場合は、「no image」という文字列を最後の行に挿入して下さい。
{additional_system_role}

プロンプトの例は以下のとおりです。

# 本文
ここに記事の本文が入ります。

# 記事のURL
ここに記事のURLが入ります。

# ハッシュタグ
ここにハッシュタグが入ります。
"""

    # Few-shots Learningによって、回答例を学習させるためのプロンプトを設定します。
    FEW_SHOTS_USER_01 = """
# 本文
AzureのApp Serviceは多機能でアプリの公開を簡単に行うことが可能。
またスケールアップ・スケールアウトによる性能調整や、Azure Active Directoryとの認証統合なども可能で、
とにかくすごい機能ばかりです。
<img decoding="async" fetchpriority="high" id="fancybox-img" class="" src="https://tech-lab.sios.jp/wp-content/uploads/2021/06/Screen-Shot-2021-06-12-at-1.37.18.png" alt="" width="601" height="283" />

# 記事のURL
tech-lab.sios.jp/archives/36497

# ハッシュタグ
#Azure #AppService
"""

    FEW_SHOTS_ASSISTANT_01 = """
AzureのApp Serviceは多機能でアプリの公開を簡単に行うことが可能。またスケールアップ・スケールアウトによる性能調整や、Azure AADとの認証統合なども可能。詳しくはこのブログを見てね! → tech-lab.sios.jp/archives/36497 #Azure #AppService
https://tech-lab.sios.jp/wp-content/uploads/2021/06/Screen-Shot-2021-06-12-at-1.37.18.png
"""

    FEW_SHOTS_USER_02 = """
# 本文
仕組みとしては、特定のメールボックスを定期的にフェッチして、そのメールアドレス宛の問い合わせ内容をデータベースに格納し、「対応中」「完了」というステータスをつけたり、その他、チケット管理に必要な色々な情報を付与します。

今回やりたいことは、アレクサに「ヘルプデスクの未処理のチケット数を教えて」と話しかけると、「12件のチケットがあります」みたいな感じで、未処理のチケット数を返してくれるスキルを開発することです。

# 記事のURL
tech-lab.sios.jp/archives/36497

# ハッシュタグ
#Azure #駆け出しエンジニアと繋がりたい
"""

    FEW_SHOTS_ASSISTANT_02 = """
アレクサをカスタマイズし、OTRSの未処理のチケット数を教えてもらう方法を学べます。Lambdaを使うことでサーバーレスを実現しています!!料金も節約できます。詳しくはこのブログを見てね! → tech-lab.sios.jp/archives/8122 #Azure #駆け出しエンジニアと繋がりたい
no image
"""

    try:
        # RSSフィードからすべての記事を取得する
        blog_url = os.environ.get("BLOG_URL")
        all_entries = get_all_entries(blog_url)

        # ランダムな記事を取得する。もし環境変数SPECIFIC_ARTICLE_URLが設定されている場合は、そのURLの記事を取得する
        specific_article_url = os.environ.get("SPECIFIC_ARTICLE_URL")
        if specific_article_url and len(specific_article_url) > 0:
            random_entry = get_specific_entry(all_entries, specific_article_url)
        else:
            random_entry = get_random_entry(all_entries)

        # 記事のURLと本文を取得する
        article_url = random_entry['link']
        max_tokens = int(os.environ.get("MAX_TOKENS", 4096)) # デフォルトは4096トークンとする
        article_content = random_entry['content']
        split_article_content = split_text_approximately(article_content, max_tokens)

        # 記事の本文からTIPSを生成する
        hash_tags = os.environ.get("HASH_TAGS", "")
        image_url, tips_without_image_url = generate_tips(split_article_content, article_url, hash_tags)

        logger.info(f"Tips generated: {tips_without_image_url}")
        logger.info(f"Image URL: {image_url}")

        # Twitter APIの認証
        consumer_key = os.environ.get("X_CONSUMER_KEY")
        consumer_secret = os.environ.get("X_CONSUMER_SECRET")
        access_token = os.environ.get("X_ACCESS_TOKEN")
        access_token_secret = os.environ.get("X_ACCESS_TOKEN_SECRET")

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth)
        client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret)

        # 画像付きのツイートを作成する
        create_tweet_with_image(api, client, tips_without_image_url, image_url)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)