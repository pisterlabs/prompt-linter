# Bilibili社交媒体网站信息收集
import feedparser
import hashlib
import json
import argparse
from datetime import datetime
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from crawlab import save_item
from kafka import KafkaProducer
from bs4 import BeautifulSoup

SUMMARIZE_PROMPT = "你需要帮用户尽可能全面地总结这些信息，这些信息之间有可能是相互不关联的(比如twitter热搜等)，也有可能是相互有关的(比如一整篇新闻报道等)，无论如何，你需要尽可能覆盖所有内容，文字可以多一些，并且以自然通顺的语言进行总结报告。"
ABSTRACT_PROMPT = "你需要使用简洁的语言总结全文摘要，尽量控制在2-3句话内。"
TAGGING_PROMPT = """你需要对这些信息打上tag，这些tag可以是文章内的关键词，也可以 是与文章有关的词汇，在tag后面，你需要添加tag在文章中的权重，权重值为在0.01-0.99之间的浮点数，tag最多不超过15个。你需要按照如下格式返回，需要保证能被json解析: 
{
    "tags": {
        "<tag1>": "<权重1>",
        "<tag2>": "<权重2>",
        ...
        "<tagN>": "<权重N>"
    }
}
"""

def ai_summarize(system_prompt, content):
    import openai
    import pymysql

    # TODO: 后续直接从IFP后端接口获取
    try:
        with pymysql.connect(
            host='192.168.238.1',
            user='root',
            password='st3zvPr4expaszYRcRgrK.!8Cb6nbbDD',
            database='info_fusion_plat'
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT value FROM platform_token WHERE env_var_name="OPENAI_API_KEY"')
                result = cursor.fetchone()[0]
                if result is None:
                    raise Exception('未找到OpenAI API Key')
                openai.api_key = result
                openai.proxy = 'http://192.168.238.1:7890'
    except Exception as e:
        print(e)
        return ''

    try:
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                },
            ], temperature=0)
    except Exception as e:
        print(e)
        return ''

    return json.loads(str(chat_completion))['choices'][0]['message']['content']



if __name__ == '__main__':
    #region 参数解析
    parser = argparse.ArgumentParser(description='Bilibili社交媒体网站信息收集')

    subparsers = parser.add_subparsers(dest='subcommand', help='类型')

    # 综合热门
    popular_parser = subparsers.add_parser('popular', help='综合热门')
    popular_parser.add_argument('--no-summarize', action='store_true', help='指定该参数时不生成总结内容')

    # 热搜
    hot_search_parser = subparsers.add_parser('hot', help='热搜')
    hot_search_parser.add_argument('--no-summarize', action='store_true', help='指定该参数时不生成总结内容')

    # 排行榜
    ranking_parser = subparsers.add_parser('ranking', help='排行榜')
    ranking_parser.add_argument('--tid', required=True, default='0', help='排行榜分区 id, 默认 0')
    ranking_parser.add_argument('--days', default='1', choices=['1', '3', '7', '30'], help='时间跨度, 可为 1 3 7 30')
    ranking_parser.add_argument('--arc-type', default='0', choices=['0', '1'], help='投稿时间, 可为 0(全部投稿) 1(近期投稿) , 默认 1')
    ranking_parser.add_argument('--disableEmbed', action='store_true', help='默认为开启内嵌视频, 任意值为关闭')
    ranking_parser.add_argument('--no-summarize', action='store_true', help='指定该参数时不生成总结内容')
    
    # 每周必看
    weekly_parser = subparsers.add_parser('weekly', help='每周必看')
    weekly_parser.add_argument('--disableEmbed', action='store_true', help='默认为开启内嵌视频, 任意值为关闭')
    weekly_parser.add_argument('--no-summarize', action='store_true', help='指定该参数时不生成总结内容')
    
    # 频道排行榜
    channel_parser = subparsers.add_parser('channel', help='频道排行榜')
    channel_parser.add_argument('--channelid', default='5417', required=True, help='必选 — 频道id, 可在频道链接中找到')
    channel_parser.add_argument('--disableEmbed', action='store_true', help='默认为开启内嵌视频, 任意值为关闭')
    channel_parser.add_argument('--no-summarize', action='store_true', help='指定该参数时不生成总结内容')

    args = parser.parse_args()
    #endregion

    # 默认综合热门
    if args.subcommand == 'hot':
        rss_url = 'http://192.168.238.128:1200/bilibili/hot-search/'
        content = "以下是{}bilibili热搜榜视频标题:<br/>".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        rss_type = 'Bilibili热搜榜'
    elif args.subcommand == 'ranking':
        rss_url = f'http://192.168.238.128:1200/bilibili/ranking/{args.tid}/{args.days}/{args.arc_type}/' + ('disableEmbed/' if args.disableEmbed else '')
        content = "以下是 {} bilibili排行榜视频标题:<br/>".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        rss_type = 'Bilibili排行榜'
    elif args.subcommand == 'weekly':
        rss_url = f'http://192.168.238.128:1200/bilibili/weekly/' + ('disableEmbed/' if args.disableEmbed else '')
        content = "以下是{}bilibili每周必看视频标题:<br/>".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        rss_type = 'Bilibili每周必看'
    elif args.subcommand == 'channel':
        rss_url = f'http://192.168.238.128:1200/bilibili/channel/{args.channelid}/' + ('disableEmbed/' if args.disableEmbed else '')
        content = "以下是{}bilibili频道排行榜视频标题:<br/>".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        rss_type = 'Bilibili频道排行榜'
    else:
        rss_url = 'http://192.168.238.128:1200/bilibili/popular/all/'
        content = "以下是{}bilibili综合热门视频标题:<br/>".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        rss_type = 'Bilibili综合热门'
    feed = feedparser.parse(rss_url)

    #region Kafka配置
    bootstrap_servers = 'kafka-server:9092'
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    topic = 'rss_handle'
    #endregion

    count = 1
    raw_content = content
    url_content = content

    for entry in feed.entries:
        raw_content += f"{count}. {entry.title}<br/>"
        url_content += f'{count}. <a href="{entry.link}" target="_blank" >{entry.title}</a><br/>'
        count+=1

    data = {}

    data['title'] = rss_type + ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        data['link'] = feed.feed.link
    except Exception as e:
        data['link'] = 'https://www.bilibili.com'

    summarize =''
    if not args.no_summarize:
        summarize = ai_summarize(SUMMARIZE_PROMPT, raw_content).replace('\n\n', '\n').replace('\n', '<br/>')
    data['post_content'] = url_content + '<br/>' + summarize
    try:
        data['post_time'] = datetime.strptime(feed.feed.updated, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        data['post_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    try:
        keywords = json.loads(ai_summarize(TAGGING_PROMPT, raw_content))
    except Exception as e:
        print('标签生成报错', str(e))
        keywords = {'标签生成错误': 0.0001}
    data['keywords'] = json.dumps(keywords.get('tags', {}), ensure_ascii=False)

    if args.no_summarize or summarize == '':
        summarize = data['post_content']
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=BeautifulSoup(summarize, 'html.parser').get_text(), lower=True, window=2)
    important_keywords = []
    for phrase in tr4w.get_keyphrases(keywords_num=5, min_occur_num=2):
        important_keywords.append(phrase)
    data['important_keywords'] = important_keywords

    data['abstract'] = ai_summarize(ABSTRACT_PROMPT, raw_content)
    data['gather_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['deduplication_id'] = hashlib.md5((str(data['post_time'])+str(data['post_content'])+str(data['title'])).encode('utf-8')).hexdigest()
    
    # other meta
    data['table_type'] = 'rss'
    data['rss_type'] = 'bilibili'
    data['platform'] = rss_type
    data['meta'] = ['低关联度内容']
    if not args.no_summarize:
        data['meta'].append('包含AI生成内容')
        data['meta'].append('谨慎判断生成内容真实性')

    #region 推送kafka
    push_kafka_success = False
    try:
        future = producer.send(topic, json.dumps(data).encode('utf-8'))
        # future.get(timeout=100)
        producer.flush()
        push_kafka_success = True
    except Exception as e:
        print('kafka推送报错', str(e))
        push_kafka_success = False
    #endregion

    data['push_kafka_success'] = push_kafka_success
    print(f'添加: {data}')
    save_item(data)

    producer.close()
