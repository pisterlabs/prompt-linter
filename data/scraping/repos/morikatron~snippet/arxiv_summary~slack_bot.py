"""
Created by Hikaru Yamada
Copyright (c) 2023 Morikatron Inc. All rights reserved.
"""

import time
import os
import re
import requests
import itertools
import json
import csv
from tqdm import tqdm
from datetime import date
from multiprocessing import Process
from typing import List

from slack_bolt import App, Say
import arxiv
import openai
from openai import error
import schedule

# for logger　--------------------
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('logfile.txt')
logger.addHandler(handler)
# --------------------for logger　


# Slack Bot用の設定
SLACK_SIGNING_SECRET = "YOUR_SLACK_SIGNING_SECRET"
SLACK_BOT_TOKEN = "YOUR_SLACK_BOT_TOKEN"
SLACK_CHANNEL = "YOUR_SLACK_CHANNEL"
PORT = 3000

# OpenAIのapiキー
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# 各種パラメータ
DEFAULT_LANGUAGE = "Japanese"  # 要約先の言語
MODEL_NAME = "gpt-3.5-turbo"  # 文章生成モデル
NUM_ARXIV_PAPERS = 2500  # arxiv APIから取得する論文情報の総数
NUM_SUMMARIZE = 3  # 要約する論文数の上限
QUERY_CATEGORY = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'stat.ML']  # カテゴリでクエリを指定する場合のカテゴリリスト
NUM_CITED_BY = 2  # 被引用数の下限
NUM_SUMMARY_FROM_QUERY = 3  # クエリから要約する論文数の上限
YEAR_AGO = 2  # 何年前までの論文を対象にするか

# 論文要約用のプロンプト
PROMPT_SUMMARIZE = """Summarise the main points of the given paper in 3 points only.
Please output the following format "MUST be in {language}". ```
{language} translation of the title
- Main point 1
- Main point 2
- Main point 3.
```"""
# クエリを作成するためのプロンプト
PROMPT_MAKE_QUERY = """Select words from the input string that you consider important for searching for articles.
Please make sure to translate all selected words into English.

The output must be in JSON format.
Example: {{"words":["Deep", "Learing"]}}
"""

# slack appを作成
app = App(signing_secret=SLACK_SIGNING_SECRET, token=SLACK_BOT_TOKEN)
openai.api_key = OPENAI_API_KEY

# arxivのバージョンを取得するための正規表現
re_version = re.compile(r'(v\d+?)$')

# 要約した論文のIDとその情報を保存するファイルを作成
summarized_file = './arxiv_summaries.tsv'
if not os.path.exists(summarized_file):
    with open(summarized_file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        header = ['arxiv_id', 'created_at', 'summary', 'num_cited_by', 'citation_velocity']
        writer.writerow(header)


# デフォルではない言語に設定する言語の人のIDとその言語
other_language = {"USER1_ID": "Chinese",
                  "USER2_ID": "English",
                  }
# set jaなどの'ja'を'Japanese'に変更するためのdict
language_abbr = {"ja": "Japanese",
                 "en": "English",
                 "ch": "Chinese",
                 }


def get_user_language():
    user_language = {}
    # ユーザー名を取得
    for user in app.client.users_list()['members']:
        if user.get('id') in other_language.keys():
            user_language[user['id']] = other_language[user['id']]
        else:
            user_language[user['id']] = DEFAULT_LANGUAGE
    return user_language


user_language = get_user_language()


def create_summarized_message(result: arxiv.arxiv.Result, language=DEFAULT_LANGUAGE, show_url=True) -> str:
    """slackに投稿するようの要約を作成する

    Parameters
    ----------
    result : arxiv.arxiv.Result
        arxiv APIから取得した論文情報結果
    show_url : bool, optional
        slackに投稿する際にURLを付属するか否か, by default True

    Returns
    -------
    str
        slackに投稿する用のメッセージ
    """
    text = f"title: {result.title}\nbody: {result.summary}"
    summary = get_response_from_gpt(system_content=PROMPT_SUMMARIZE.format(language=language), user_content=text)
    title_en = result.title
    title, *body = summary.split('\n')
    body = '\n'.join(body)
    date_str = result.published.strftime("%Y-%m-%d %H:%M:%S")
    message = ""
    message += f"{title_en}\n"
    if title_en != title:
        message += f"{title}\n"
    message += f"{body}\n"
    message += f"Published: {date_str}\n"
    if show_url:
        message += f"{result.entry_id}"
    return message


def post_summary() -> None:
    """要約をSlackに投稿する"""
    # すでに要約した論文のIDを取得
    with open(summarized_file) as f:
        reader = csv.reader(f, delimiter='\t')
        already_summarized = [r[0] for r in reader]
    # 今日の日付
    today_date = date.today().isoformat()

    """ arxiv APIを使い、検索クエリに応じた結果（search）をdictに格納"""
    result_dict = {}
    for i, q in enumerate(tqdm(QUERY_CATEGORY, desc='arxiv search')):
        if i > 0:
            # API制限のため5秒待つ
            print('wait 5 seconds because of API limitation')
            time.sleep(5)
        query = f'cat:%22 {q} %22'  # カテゴリごとにクエリを作成
        # arxiv APIで最新の論文情報を取得する
        search = arxiv.Search(
            query=query,  # 検索クエリ（
            max_results=NUM_ARXIV_PAPERS // len(QUERY_CATEGORY),  # 取得する論文数
            sort_by=arxiv.SortCriterion.SubmittedDate,  # 論文を投稿された日付でソートする
            sort_order=arxiv.SortOrder.Descending,  # 新しい論文から順に取得する
        )
        for result in search.results():
            result_dict[result.entry_id] = result

    """要約候補の論文情報をsemantic scholarで取得"""
    summary_candidates = {}
    for i, result in enumerate(tqdm(result_dict.values(), desc='request to semantic scholar')):
        arxiv_url = result.entry_id
        if arxiv_url:
            arxiv_url = re_version.sub('', arxiv_url)
            arxiv_id = arxiv_url.split('/')[-1]
            try:
                sem = requests.get("https://api.semanticscholar.org/v1/paper/arxiv:" + arxiv_id).json()
            except Exception as e:
                logger.error('Error while requesting to semantic scholar')
                logger.error(e)
                continue
            citation_velocity = sem.get('citationVelocity', 0)  # 研究の注目度
            num_cited_by = sem.get('numCitedBy', 0)  # 被引用数
            # 新しい論文でcitation_velocityは1以上あれば十分
            if arxiv_id not in already_summarized:
                if (sem.get('year', 0) >= int(today_date[:4])) and ((num_cited_by >= NUM_CITED_BY) or (citation_velocity > 0)):
                    summary_candidates[arxiv_id] = {'result': result,
                                                    'num_cited_by': num_cited_by,
                                                    'citation_velocity': citation_velocity
                                                    }

        # 100リクエスト毎に５分待つ必要あり。（APIのLimitation)
        if i > 0 and (i % 99 == 0):
            print('wait 5 minutes because of API limitation')
            time.sleep(5 * 60)

    """要約候補の論文を要約して、slackに投稿する"""
    if len(summary_candidates) > NUM_SUMMARIZE:
        summary_candidates = {k: v for k, v in sorted(
            summary_candidates.items(), key=lambda x: x[1]['num_cited_by'], reverse=True)[:NUM_SUMMARIZE]}
    with open(summarized_file, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for i, (arxiv_id, v) in enumerate(tqdm(summary_candidates.items(), desc='summarize arxiv papers')):
            # 二度同じ論文を要約しないように保存
            # Slackに投稿するメッセージを組み立てる
            print(f"要約start: 論文{i+1}本目")
            summary = create_summarized_message(v['result'])
            if summary:
                message = f"{today_date}の論文です！ " + str(i + 1) + "本目\n" + f"被引用数: {v['num_cited_by']}\n" + summary
                # Slackにメッセージを投稿する
                app.client.chat_postMessage(text=message, channel=SLACK_CHANNEL)
                print(f"要約投稿完了: 論文{i+1}本目")
                writer.writerow([str(arxiv_id), str(today_date), str(summary.replace('\t', ' ')), str(
                    v['num_cited_by']), str(v['citation_velocity'])])


@app.event({"type": "message", "subtype": None})
def reply_message(body: dict, say: Say) -> None:
    """slackでユーザーがなにか入力した時に呼ばれる関数"""
    global user_language
    event = body["event"]

    # ユーザのidや発言内容などを取得
    user_id = event.get('user')
    channel_token = event.get('channel')  # 'XXXXXXXXXX'など
    user_msg_text = event.get('text')  # slackに入力された文字列
    print('user_msg_text: ', user_msg_text)

    if user_msg_text[0] in {'#', '＃'}:  # コメントが入力された時
        print('comment inputted')
        return

    if user_msg_text[:3] == 'set':
        changed_language = user_msg_text[-2:]
        if changed_language not in language_abbr.keys():
            res = f"{list(language_abbr.keys())}の中から言語を選んでください"
        else:
            user_language[user_id] = changed_language
            res = f"要約される言語が{language_abbr[changed_language]}に変更されました"
        res = add_mention(res, user_id)
        say(res, channel=channel_token)
        return

    if user_msg_text[1:5] == 'http':  # arxivのURLが入力された時
        arxiv_url = user_msg_text[1:-1]
        arxiv_url = re_version.sub('', arxiv_url)
        arxiv_id = arxiv_url.split('/')[-1]
        if arxiv_id[-4:] == '.pdf':
            arxiv_id = arxiv_id[:-4]
        arxiv_ids = [arxiv_id]
        try:
            search = arxiv.Search(
                id_list=arxiv_ids
            )
            results = [r for r in search.results()]
        except Exception as e:
            logger.error(e)
            say("arxivのURLが見つかりません", channel=channel_token)
            return
        if not results:
            say("arxivのURLが見つかりません", channel=channel_token)
            return
        summary = create_summarized_message(results[0], show_url=False, language=user_language[user_id])
        say(summary, channel=channel_token)

    else:  # URL以外の文字列が入力された時、GPTでクエリを生成する
        response_content = get_response_from_gpt(system_content=PROMPT_MAKE_QUERY, user_content=user_msg_text)
        debug_comment = "クエリからうまく論文が見つけられませんでした。クエリを直してもう一度入力してください。"
        try:
            json_text = json.loads(response_content)
            query = ' '.join(json_text.get('words', []))
        except Exception as e:
            logger.error(e)
            say(debug_comment, channel=channel_token)
            return
        arxiv_ids = get_arxiv_ids_from_query(query)
        try:
            search = arxiv.Search(
                id_list=arxiv_ids
            )
            results = [r for r in search.results()]
        except Exception as e:
            logger.error(e)
            say(debug_comment, channel=channel_token)
            return
        if (not results) or (not arxiv_ids):
            say(debug_comment, channel=channel_token)
            return
        is_query_summarized = False
        say(f"【使用クエリ】 {query}", channel=channel_token)
        for i, result in enumerate(results[:NUM_SUMMARY_FROM_QUERY]):
            summary = create_summarized_message(result, show_url=True, language=user_language[user_id])
            send_text = ""
            if summary:
                is_query_summarized = True
                send_text = f"【{i + 1}本目の候補論文】\n{summary}"
                say(send_text, channel=channel_token, unfurl_links=True)
        if not is_query_summarized:
            say("クエリからうまく論文が見つけられませんでした。クエリを変えてもう一度入力してください。", channel=channel_token)
            return


def get_arxiv_ids_from_query(query: str) -> List[str]:
    """クエリからそれに応じたarxivの論文のidを取得する

    Parameters
    ----------
    query : str
        arxiv API用の検索クエリ

    Returns
    -------
    List[str]
        arxivの論文のidのリスト
    """
    today_date = date.today().isoformat()
    result_dict = {}
    search = arxiv.Search(
        query=query,  # 検索クエリ（
        max_results=100,  # 取得する論文数
        sort_by=arxiv.SortCriterion.Relevance,  # 論文を関連度でソートする
        sort_order=arxiv.SortOrder.Descending,  # 新しい論文から順に取得する
    )
    for result in search.results():
        result_dict[result.entry_id] = result

    summary_candidates = {}
    for result in tqdm(result_dict.values(), desc='request to semantic scholar'):
        arxiv_url = result.entry_id
        if arxiv_url:
            arxiv_url = re_version.sub('', arxiv_url)
            arxiv_id = arxiv_url.split('/')[-1]
            try:
                sem = requests.get("https://api.semanticscholar.org/v1/paper/arxiv:" + arxiv_id).json()
            except Exception as e:
                logger.error('while request to semantic scholar')
                logger.error(e)
                continue
            # citation_velocity = sem.get('citationVelocity', 0)
            num_cited_by = sem.get('numCitedBy', 0)
            if (sem.get('year', 0) >= int(today_date[:4]) - YEAR_AGO):  # YEAR_AGOまでの論文のみを対象とする
                summary_candidates[arxiv_id] = {'result': result,
                                                'num_cited_by': num_cited_by,
                                                # 'citation_velocity': citation_velocity
                                                }
    sorted_summary_candidates = sorted(summary_candidates.items(), key=lambda x: x[1]['num_cited_by'], reverse=True)
    sorted_candidate_arxiv_ids = [s[0] for s in sorted_summary_candidates]
    return sorted_candidate_arxiv_ids


def get_response_from_gpt(system_content: str, user_content: str) -> str:
    """GPTで応答を生成する

    Parameters
    ----------
    system_content : str
        system role用のプロンプト
    user_content : str
        user role用のプロンプト

    Returns
    -------
    str
        GPTの応答
    """
    response = {}
    for i in itertools.count():
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': user_content}
                ],
                temperature=0.25,
                timeout=60,
            )
        except error.APIConnectionError as e:
            # RemoteDisconnected対策
            logging.error(e)
            if i < 3:
                # 3回までリトライする
                time.sleep(4)
                logging.error('retry')
                continue
        except Exception as e:
            logging.error(e)
        else:
            break
    if not response:
        return ""
    return response['choices'][0]['message']['content']


@app.event({"type": "message"})
def on_else(body: dict) -> None:
    # その他のメッセージイベントを受けます（下記がない場合slackからの送信が404になり、イベントが延々再送されます）
    event = body["event"]
    print('その他のメッセージイベント:', event)


def add_mention(text: str, user_id: str) -> str:
    return '<@' + user_id + '>: \n' + text


def scheduler_post() -> None:
    """定期的に投稿する"""
    schedule.every().day.at("06:00").do(post_summary)
    while True:
        schedule.run_pending()
        time.sleep(1)


def slack_app_start() -> None:
    """slack appを起動する"""
    print('app start')
    app.start(port=PORT)


if __name__ == "__main__":
    p1 = Process(target=scheduler_post)
    p2 = Process(target=slack_app_start)
    p1.start()
    p2.start()
