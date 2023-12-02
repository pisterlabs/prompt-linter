import argparse
import datetime
import os
import time
import urllib.parse
import warnings
from dataclasses import dataclass

from make_slide import make_slides
import arxiv
import openai
from slack_sdk import WebClient
from io import BytesIO

import yaml
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from pathlib import Path

# setting
warnings.filterwarnings("ignore")

@dataclass
class Result:
    score: float = 0.0
    hit_keywords: list = None
    arxiv_result: dict = None
    abst_jp: str = None


PROMPT = """与えられた論文の要点をまとめ、以下の項目で日本語で出力せよ。それぞれの項目は最大でも180文字以内に要約せよ。
```
論文名:タイトルの日本語訳
キーワード:この論文のキーワード
課題:この論文が解決する課題
手法:この論文が提案する手法
結果:提案手法によって得られた結果
```"""
BASE_DIR=Path("./files")
CHANNEL_ID = "C03KGQE0FT6"


def calc_score(abst: str, keywords: dict):
    sum_score = 0.0
    hit_kwd_list = []

    for word in keywords.keys():
        score = keywords[word]
        if word.lower() in abst.lower():
            sum_score += score
            hit_kwd_list.append(word)
    return sum_score, hit_kwd_list


def get_text_from_driver(driver) -> str:
    try:
        # elem = driver.find_element_by_class_name("lmt__translations_as_text__text_btn")
        elem = driver.find_element(by=By.CLASS_NAME, value="lmt__translations_as_text__text_btn")
    except NoSuchElementException as e:
        return None
    text = elem.get_attribute("innerHTML")
    return text


def get_translated_text(from_lang: str, to_lang: str, from_text: str, driver) -> str:
    sleep_time = 1
    from_text = urllib.parse.quote(from_text)
    url = "https://www.deepl.com/translator#" \
        + from_lang + "/" + to_lang + "/" + from_text

    driver.get(url)
    driver.implicitly_wait(10)

    for i in range(30):
        time.sleep(sleep_time)
        to_text = get_text_from_driver(driver)
        if to_text:
            break
    if to_text is None:
        return urllib.parse.unquote(from_text)
    return to_text


def search_keyword(
        articles: list, keywords: dict, score_threshold: float
        ):
    options = webdriver.FirefoxOptions()
    options.headless = True
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
    results = []
    for article in articles:
        abstract = article.summary.replace("\n", " ")
        score, hit_keywords = calc_score(abstract, keywords)
        if score < score_threshold:
            continue
        abstract_trans = get_translated_text("ja", "en", abstract, driver)

        result = Result(score=score, hit_keywords=hit_keywords, arxiv_result=article, abst_jp=abstract_trans)
        results.append(result)

    driver.quit()
    return results


def get_summary(result):
    title = result.title.replace("\n ", "")
    body = result.summary.replace("\n", " ")
    text = f"title: {title}\nbody: {body}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.25,
    )
    summary = response["choices"][0]["message"]["content"]
    summary_dict = {}
    for b in summary.split("\n"):
        if b.startswith("論文名"):
            summary_dict["title_jp"] = b[4:].lstrip()
        if b.startswith("キーワード"):
            summary_dict["keywords"] = b[6:].lstrip()
        if b.startswith("課題"):
            summary_dict["problem"] = b[3:].lstrip()
        if b.startswith("手法"):
            summary_dict["method"] = b[3:].lstrip()
        if b.startswith("結果"):
            summary_dict["result"] = b[3:].lstrip()
        summary_dict["title"]= result.title

    summary_dict["id"] = result.get_short_id().replace(".", "_")
    summary_dict["date"] = result.published.strftime("%Y-%m-%d %H:%M:%S")
    summary_dict["authors"] = result.authors
    summary_dict["year"] = str(result.published.year)
    summary_dict["entry_id"] = str(result.entry_id)
    summary_dict["primary_category"] = str(result.primary_category)
    summary_dict["categories"] = result.categories
    summary_dict["journal_ref"] = result.journal_ref
    summary_dict["pdf_url"] = result.pdf_url
    summary_dict["doi"]= result.doi
    summary_dict["abstract"] = body
    return summary_dict


def send2app(text: str, slack_token: str, file: str=None) -> None:
    if slack_token is not None:
        client = WebClient(token=slack_token)
        if file is None:
            new_message = client.chat_postMessage(
                channel=CHANNEL_ID,
                text=text,
            )
        else:
            print(file)
            with open(file, "rb") as f:
                new_file = client.files_upload(
                    channels=CHANNEL_ID,
                    file=BytesIO(f.read()),
                    filename=file.name,
                    filetype="pdf",
                    initial_comment=text,
                )


def notify(results: list, slack_token: str) -> None:
    star = "*"*80
    today = datetime.date.today()
    n_articles = len(results)
    text = f"{star}\n \t \t {today}\tnum of articles = {n_articles}\n{star}"
    send2app(text, slack_token)
    for result in sorted(results, reverse=True, key=lambda x: x.score):
        ar = result.arxiv_result
        url = ar.entry_id
        title = ar.title.replace("\n ", "")
        word = result.hit_keywords
        score = result.score
        abstract = result.abst_jp.replace("。", "。\n>")
        if abstract[-1] == "\n>":
            abstract = abstract.rstrip("\n>")
        abstract_en = ar.summary.replace("\n", " ").replace(". ", ". \n>")
        
        text = f"\n Score: `{score}`"\
               f"\n Hit keywords: `{word}`"\
               f"\n URL: {url}"\
               f"\n Title: {title}"\
               f"\n Abstract:"\
               f"\n>{abstract}"\
               f"\n Original:"\
               f"\n>{abstract_en}"\
               f"\n {star}"
        
        file = None
        if openai.api_key is not None:
            try:
                summary_dict = get_summary(ar)
                summary_dict["abst_jp"] = result.abst_jp
                id = summary_dict["id"]
                dirpath = BASE_DIR/id
                dirpath.mkdir(parents=True, exist_ok=True)
                pdf = f"{id}.pdf"
                ar.download_pdf(dirpath=str(dirpath), filename=pdf)
                summary_dict["pdf"] = str(dirpath/pdf)
                file = make_slides(dirpath, id, summary_dict)
            except Exception as e:
                print(e)
        send2app(text, slack_token, file)

def get_config():
    file_abs_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_abs_path)
    config_path = f"{file_dir}/../config.yaml"
    with open(config_path, "r", encoding="utf-8") as yml:
        config = yaml.load(yml)
    return config


def main():
    # debug用
    parser = argparse.ArgumentParser()
    parser.add_argument("--slack_token", default=None)
    parser.add_argument("--openai_api", default=None)
    args = parser.parse_args()

    config = get_config()
    subject = config["subject"]
    keywords = config["keywords"]
    score_threshold = float(config["score_threshold"])

    day_before_yesterday = datetime.datetime.today() - datetime.timedelta(days=2)
    day_before_yesterday_str = day_before_yesterday.strftime("%Y%m%d")
    arxiv_query = f"({subject}) AND " \
                  f"submittedDate:" \
                  f"[{day_before_yesterday_str}000000 TO {day_before_yesterday_str}235959]"
    articles = arxiv.Search(query=arxiv_query,
                           max_results=1000,
                           sort_by = arxiv.SortCriterion.SubmittedDate).results()
    articles = list(articles)
    openai.api_key = os.getenv("OPENAI_API") or args.openai_api
    results = search_keyword(articles, keywords, score_threshold)
    slack_token = os.getenv("SLACK_BOT_TOKEN") or args.slack_token
    notify(results[:1], slack_token)


if __name__ == "__main__":
    main()
