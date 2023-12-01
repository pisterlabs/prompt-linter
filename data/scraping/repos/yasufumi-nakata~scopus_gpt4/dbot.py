from discord.ext import commands
import os
import requests
import openai
import random
import discord

import asyncio
import datetime
import pytz

import functools
import typing

TOKEN = ""
openai.api_key = ""
ELSEVIER_API_KEY = ""
# CHANNEL_ID = int(os.getenv('CHANNEL_ID3'))

HEADERS = {
    'X-ELS-APIKey': ELSEVIER_API_KEY,
    'Accept': 'application/json',
    'x-els-resourceVersion': 'XOCS'
}

BASE_URL = 'http://api.elsevier.com/content/search/scopus?'

ABSTRACT_BASE_URL = 'https://api.elsevier.com/content/abstract/eid/'


def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper


@to_thread
def get_summary(result):
    print("start searching")
    eid = result.get('eid', None)
    abstract = get_abstract(eid) if eid else None
    description = abstract if abstract else result.get(
        'dc:description', 'No description available')
    text = f"タイトル: {result['dc:title']}\n内容: {description}"
    system = """与えられた論文の要点を以下のフォーマットで日本語で出力してください。タイトルは**で囲み，本文は``で囲んでください，```
         **タイトルの日本語訳**
         ``・どんなもの?``
         ``・先行研究と比べてどこがすごい?``
         ``・技術や手法のキモはどこ?``
         ``・どうやって有効だと検証した?``
         ``・議論はある?``
         ```"""

    print(text)
    print("waiting openai...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': text}
        ],
        temperature=0.7,
    )
    print("response is ready")
    summary = response['choices'][0]['message']['content']
    title_en = result['dc:title']
    title, *body = summary.split('\n')
    body = '\n'.join(body)
    date_str = result['prism:coverDate']
    link = result['link'][2]['@href']
    message = f"発行日: {date_str}\n{link}\n{title_en}\n{title}\n{body}\n"

    print("message is ready")
    return message


def get_abstract(eid):
    abstract_url = f"{ABSTRACT_BASE_URL}{eid}"
    response = requests.get(abstract_url, headers=HEADERS)
    response.raise_for_status()
    abstract_data = response.json()

    if 'abstracts-retrieval-response' in abstract_data:
        coredata = abstract_data['abstracts-retrieval-response'].get(
            'coredata', None)
        if coredata:
            return coredata.get('dc:description', None)

    return None


# インテントの生成
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents)


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')


@bot.command()
async def query(ctx, *args):
    if ctx.author == bot.user:
        return
    else:
        query_arg = ' AND '.join(args)
        query = f'TITLE-ABS-KEY({query_arg})'
        print(f"query was set:{query}")

        # Elsevier APIで最新の論文情報を取得する
        search_url = f"{BASE_URL}query={query}&count=100&sort=-date&view=STANDARD"

        response = requests.get(search_url, headers=HEADERS)
        # Add this line to raise an exception if there's an HTTP error
        response.raise_for_status()
        search_results = response.json()

        print("search done")
        # searchの結果をリストに格納
        if 'search-results' in search_results and 'entry' in search_results['search-results']:
            result_list = search_results['search-results']['entry']
        else:
            print("Error: Unexpected API response")
            result_list = []

        # ランダムにnum_papersの数だけ選ぶ
        num_papers = 1
        num_papers = min(num_papers, len(result_list))
        results = random.sample(result_list, k=num_papers)
        print("results are ready")

        # 論文情報をDiscordに投稿する
        for i, result in enumerate(results):
            print(f"Processing result {i+1}" + "／" + str(num_papers))
            await ctx.channel.send(f"Processing result {i+1}" + "／" + str(num_papers) + " query: " + str({query_arg}))
            try:
                msg = str(query_arg)+"の論文です！ " + str(i+1) + "／" + str(num_papers) + await get_summary(result)
                print(f"{msg}")
                print("done!")
                await ctx.channel.send(msg)
            except:
                print("Error posting message")
                print(await get_summary(result))
print("bot run")
bot.run("")
