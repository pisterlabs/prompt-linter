# load api keys, libraries
import pandas as pd
import numpy as np
import json
import openai
import pinecone
import csv
import os
import time
from datetime import datetime, timedelta
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# set working directory to the folder with the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

#load openai & pinecone credentials
with open('keys/api_keys.json') as f:
  credentials = json.loads(f.read())

openai_key = credentials['openai_key']
pine_key = credentials['pine_key']
pine_env = credentials['pine_env']
tg_token = credentials['tg_token']

# init openai & pinecone
openai.api_key = openai_key

# CLOUD init for PINECONE due to proxy restrictions of pythonanywhere. Openai config skipped - pinecone do not connect to openai.
# from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
# openapi_config = OpenApiConfiguration.get_default_copy()
# openapi_config.proxy = "http://proxy.server:3128"
# pinecone.init(
#         api_key=pine_key,
#         environment=pine_env,
#         openapi_config=openapi_config
#     )
# DEFAULT LOCAL init for PINECONE
pinecone.init(api_key=pine_key, environment=pine_env)
index_name = 'tg-news'
index = pinecone.Index(index_name)

# global variables
request = "" # question to ask media
dates = None # list of min and max dates. If only one date - assumed to be min date.
sources = [] # list of sources. Must be exact match to the source name in the dataset
stance = [] # list of stances. Must be exact match to the content type in the dataset ['inet propaganda', 'altern', 'tv', 'moder', 'voenkor']

model_name = "gpt-3.5-turbo" # model for summarization
# better model can accomodate more newsю, however, the effect of more news in context should be tested

# ======================== #
# FUNCTIONS FOR MEDIA REQUEST #
# ======================== #

# models pricing: https://openai.com/pricing
def get_price_per_1K(model_name):
    if model_name == "gpt-3.5-turbo": #4K (~10 news)
        price_1K = 0.0015 # price per 1000 characters
    elif model_name == "gpt-3.5-turbo-16k": #16K (~40 news)
        price_1K = 0.003
    elif model_name == "gpt-4": #8K (~20 news)
        price_1K = 0.03
    elif model_name == "gpt-4-32k": #32K (~80 news)
        price_1K = 0.06
    return price_1K

price_1K = get_price_per_1K(model_name)


## find top relevant news
# embed request
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# get similar news from PINECONE with filters (dates=None, sources=None, stance=None)
@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=10))
def get_top_pine(request=None, request_emb=None, dates=None, sources=None, stance=None, model="text-embedding-ada-002", top_n=10):
    """
    Returns top news articles related to a given request and stance, within a specified date range.

    Args:
        request (str): The request for which to find related news articles.
        request_emb (numpy.ndarray): The embedding of the request, if already computed.
        dates (list): A list of one or two dates in the format 'YYYY-MM-DD', representing the start and end dates of the date range to search for news articles. If only one date is provided, the end date will be set to today.
        stance (str): The stance of the news articles to search for. Must be one of 'positive', 'negative', or 'neutral'.
        model (str): The name of the OpenAI model to use for computing embeddings.
        top_n (int): The number of top news articles to return.

    Returns:
        Tuple of two strings:
        - The first string contains the summaries of the top news articles.
        - The second string contains the links to the top news articles, along with their similarity scores.
    """
    if request_emb is None and request is None:
        print('Error: no request')
        return
    if request_emb is None:
        request_emb = get_embedding(request)

    dates=dates
    stance=stance[0]
    # define start and end dates (if end date is not defined, it will be set to today)
    if dates:
        if len(dates) == 2:
            # convert start_date to int
            start_date = dates[0]
            end_date = dates[1]
        else:
            start_date = dates[0]
            end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        # set range from 2022-02-01 to today
        start_date = '2000-02-01'
        end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

    # filtering
    start_date = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_date = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

    filter = {
        "stance": { "$eq": stance },
        "date": { "$gte": start_date, "$lte": end_date }
        }

    # query pinecone
    res = index.query(request_emb, top_k=10, include_metadata=True, filter=filter)
    #save results to txt-file
    with open('pinecone_results.txt', 'w') as f:
        f.write(str(res.to_dict()))
    # check if results are empty
    if res.to_dict()['matches'] == []:
        print('No matches')
        return 'No matches', 'No matches'
    top_sim_news = pd.DataFrame(res.to_dict()['matches']).join(pd.DataFrame(res.to_dict()['matches'])['metadata'].apply(pd.Series))

    # collect links & similarities
    top_sim_news['msg_id'] = top_sim_news['id'].apply(lambda x: x.split('_')[-1])
    top_sim_news['channel_name'] = top_sim_news['id'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    top_sim_news['link'] = top_sim_news.apply(lambda x: "https://t.me/"+str(x.channel_name)+"/"+str(x.msg_id)+" - "+str(round(x.score,3)), axis=1)
    news_links = '\n'.join(top_sim_news['link'].tolist())
    # collect news
    news4request = '\n'.join(top_sim_news['summary'].tolist())
    return news4request, news_links

## Ask OpenAI
@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=10))
def ask_openai(request, news4request, model_name = "gpt-3.5-turbo", tokens_out = 512):

    system_content_en = f"You are given few short news texts in Russian. Based on these texts you need to answer the following question: {request}. \
        First, analyze if the texts provide an answer to the question. \
        If the texts do not provide proper answer, say that. \
        If they do, select the texts relevant to the question ({request}) and summarize them. \
        \nОтвечай только на русском. Не более 1000 символов."

    system_content_ru = f"Тебе будут представлены несколько новостей. На их основе нужно ответить на вопрос: {request}. \
        Сперрва проверь, что новости содержат ответ. \
        Если ответа в новостях нет, так и ответь. \
        Далее, отбери новости, которые отвечают на вопрос ({request}) и сделай но ним резюме. \
        \nНе более 1000 символов."

    response = openai.ChatCompletion.create(
        model = model_name,
        messages=[
            {
            "role": "system",
            "content": system_content_ru
            },
            {
            "role": "user",
            "content": news4request
            }
        ],
        temperature=0,
        max_tokens=tokens_out,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    return response

# FUNCTION ask_media to combine all together (TO USE IN TG BOT REQUESTS)
def ask_media(request, dates=None, sources=None, stance=None, model_name = "gpt-3.5-turbo", tokens_out = 512, full_reply = True):
    # check request time
    request_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # get top news
    # INPUT: request, dates, sources, stance. OUTPUT: news4request - list of news texts for openai, news_links - list of links
    news4request, news_links = get_top_pine(request, dates=dates, sources=sources, stance=stance, model="text-embedding-ada-002", top_n=10)
    if news4request == 'No matches':
        return "No matches. Seems filter is too strict."
    # limit number of tokens vs model
    if model_name == "gpt-3.5-turbo":
        # print(type(news4request), len(news4request))
        # print(news4request)
        news4request = news4request[:4000]
    elif model_name == "gpt-3.5-turbo-16k":
        news4request = news4request[:16000]
    elif model_name == "gpt-4":
        news4request = news4request[:8000]
    elif model_name == "gpt-4-32k":
        news4request = news4request[:32000]

    reply = ask_openai(request, news4request, model_name = model_name, tokens_out = tokens_out)
    request_params = f"Request: {request}; \nFilters: dates: {dates}; sources: {sources}; stance: {stance}"
    reply_text = reply.choices[0]['message']['content']
    n_tokens_used = reply.usage.total_tokens
    reply_cost = n_tokens_used / 1000 * price_1K

    # write params & reply to file. If file doesn't exist - create it with headers
    # check reply time
    reply_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if not os.path.isfile('openai_chatbot.csv'):
        with open('openai_chatbot.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['request', 'dates', 'sources', 'stance', 'reply_text', 'reply_cost', 'request_time', 'reply_time', 'model_name', 'n_tokens_used', 'news_links'])
    with open('openai_chatbot.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([request, dates, sources, stance, reply_text, reply_cost, request_time, reply_time, model_name, n_tokens_used, news_links])

    # return reply for chatbot. If full_reply = False - return only reply_text
    if full_reply == False:
        return reply_text
    else:
        return request_params + "\n" + "Cost per request: " + str(round(reply_cost,3)) + ". Tokens used: " + str(n_tokens_used) + "\n\n" + reply_text + "\n\n" + news_links

def compare_stances(request, summaries_list, model_name = "gpt-3.5-turbo", tokens_out = 1500):
    # check request time
    request_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    system_content_en = f"You are given few texts in Russian from 5 sources on the same subject: {request} \
The structure of the texts is as follows:\
1) name of the source is given in []\
2) the text on the subject above is given.\
You task is to analyse what is similar and what is different in all these texts. First, tell what is similar. Then tell the differences for each source. \
\nОтвечай только на русском. "

    system_content_ru = f"Тебе будут представлены несколько текстов из источников на одну тему: {request} \
Структура текстов следующая:\
1) в [] указан источник\
2) далее идет текст на тему выше.\
Твоя задача - проанализировать, что общего и что разного в этих текстах. Сначала скажи, что общего. \n\
Затем, для каждого источника, скажи, в чем разница в формате: [истчоник] - в чем отличия.\
\nВсего не более 1500 символов."

    reply = openai.ChatCompletion.create(
    model = model_name,
    messages=[
            {
            "role": "system",
            "content": system_content_ru
            },
            {
            "role": "user",
            "content": summaries_list
            }
        ],
        temperature=0,
        max_tokens=tokens_out,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    request_params = f"Request: {request}; \nFilters: dates: {dates}; sources: {sources}; stance: {stance}"
    reply_text = reply.choices[0]['message']['content']
    n_tokens_used = reply.usage.total_tokens
    reply_cost = n_tokens_used / 1000 * price_1K
    reply_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if not os.path.isfile('openai_chatbot.csv'):
        with open('openai_chatbot.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['request', 'dates', 'sources', 'stance', 'reply_text', 'reply_cost', 'request_time', 'reply_time', 'model_name', 'n_tokens_used', 'news_links'])
    with open('openai_chatbot.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([request, dates, sources, 'all_summary', reply_text, reply_cost, request_time, reply_time, model_name, n_tokens_used, ""])

    return request_params + "\n" + "Cost per request: " + str(round(reply_cost,3)) + ". Tokens used: " + str(n_tokens_used) + "\n\n" + reply_text



# ======================== #
# FUNCTIONS FOR TELEGRAM BOT #
# ======================== #

#=== TECHNICAL & INFO functions ===#

async def start_ru(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=\
                                   "Чтобы отфильтровать новости с определенной дате, введите /set_date ГГГГ-ММ-ДД (если набрать 2 даты, то вторая дата - окончание периода).\n \
Чтобы проверить даты, введите /show_dates.\n \
Чтобы спросить у какой-то группы источников, введите /ask_voenkor, /ask_alter, /ask_moder, /ask_inet_prop или /ask_tv + ваш вопрос.\n \
Чтобы сравнить группы источников между собой, введите /ask_all + ваш вопрос.\n\
Важные моменты:\n\
1: Сейчас база содержит только новости по войне в Украине с 2022-02-01 по 2023-01-31.\n\
2: Модель отвечает на основе наиболее близких 10 новостей, поэтому не дает полной картины.\n\
3: Как и другие LLM, может галлюцинировать. Но вы можете проверить сами новости, пройдя по ссылкам.\n\
4: Запросы лучше делать в виде вопросов (желательно конкретнее)), а не общих тем (так настроены промпты). Например, 'какова ситуация с мирными жителями в Украине?' вместо 'мирные жители в Украине'")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=\
                                   "To set start date for news filtering, type /set_date YYYY-MM-DD (if 2 dates are typed, second is set as finish date). \
To show dates, type /show_dates.\n \
To ask specific media stance, type /ask_voenkor, /ask_alter, /ask_moder, /ask_inet_prop or /ask_tv + your question.\n \
To ask & compare all stances, type /ask_all + your question.\n\
To get info about the stances, type /info.\n\
Important notes:\n\
Note1: Database is limited to news from roughly the past month from 40+ different Telegram channels\n\
Note2: Answers are based on sample of 10 news, so it's not a comprehensive overview.\n\
Note3: Like all LLM it may sometimes hallucinate. But you can check the refered news following the links\
    ")

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=\
                                   "We use Telegram channels as sources of news because \n\
1) Almost all of news sources have TG channels \n\
2) News in TG channels are already shortned which facilitates AI processing \n\
Here is the desription of stances:\n\
[voenkor] - military correspondents\n\
[tv] - telegram  TV channels\n\
[inet propaganda] - pro-governent internet media (RIA Novosty, Interfax, AIF, SolovievLive, etc.)\n\
[moder] - moderate media, mostly have been independent years ago (including RBC, Kommersant, BFM, Lenta, etc.) \n\
[altern] - independent media, mostly labeled Foreign Agents by Russian authorities (including Meduza, BBC Russian, TheBell, etc.)\n\
    ")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply = update.message.text
    # await context.bot.send_message(chat_id=update.effective_chat.id, text=f"reply - {reply}")
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"I understand only commands. Type /start to see possible commands.")

# async def caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     text_caps = ' '.join(context.args).upper()
#     await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

# async def caps1(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     text_caps = context.args.append(date)
#     await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global model_name, price_1K
    model_name = context.args[0]
    # check if model is valid
    if model_name not in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"{model_name} - Invalid model name. Available models: gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-4, gpt-4-32k")
    else:
        price_1K = get_price_per_1K(model_name)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Model for summarization set to: "+model_name)

async def show_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Model for summarization: "+model_name)

#====DATES functions====#
async def set_date(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global dates
    reply = context.args
    #check if date(s) are valid
    print(f"checking {reply}")
    if len(reply) > 2 or len(reply) == 0:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"{reply} - {len(reply)}. Either one (start_date) or two (start_date, finish_date) are allowed.")
    elif len(reply) == 1:
        dates = [reply[0]]
        try:
            pd.to_datetime(dates)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Now start date for filter = "+dates[0])
        except ValueError:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Invalid date. Format should be YYYY-MM-DD, no commas, no quotes")
    elif len(reply) == 2:
        dates = [reply[0], reply[1]]
        try:
            pd.to_datetime(dates)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Now start date for filter = {dates[0]} and finish date = {dates[1]}")
        except ValueError:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Invalid date. Format should be YYYY-MM-DD, no commas, no quotes")

async def show_dates(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if dates == None:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No dates set. Type /set_date YYYY-MM-DD (if 2 dates are typed, second is set as finish date).")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="start_date:"+str(dates))

async def reset_dates(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global dates
    dates = None
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Dates reset. Now all dates of database are used.")

#=== ASK_MDEDIA functions ===#
#/ask_voenkor, /ask_alter, /ask_moder, /ask_inet_prop or /ask_tv || ['inet propaganda', 'altern', 'tv', 'moder', 'voenkor']
async def ask_voenkor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    request = ' '.join(context.args)
    if request == '':
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No request")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="In progress...Usually takes 10-20 sec")
        reply = ask_media(request, dates=dates, stance=['voenkor'], model_name = model_name)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)

async def ask_alter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    request = ' '.join(context.args)
    if request == '':
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No request")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="In progress...Usually takes 10-20 sec")
        reply = ask_media(request, dates=dates, stance=['altern'], model_name = model_name)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)

async def ask_inet_prop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    request = ' '.join(context.args)
    if request == '':
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No request")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="In progress...Usually takes 10-20 sec")
        reply = ask_media(request, dates=dates, stance=['inet propaganda'], model_name = model_name)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)

async def ask_moder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    request = ' '.join(context.args)
    if request == '':
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No request")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="In progress...Usually takes 10-20 sec")
        reply = ask_media(request, dates=dates, stance=['moder'], model_name = model_name)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)

async def ask_tv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    request = ' '.join(context.args)
    if request == '':
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No request")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="In progress...Usually takes 10-20 sec")
        reply = ask_media(request, dates=dates, stance=['tv'], model_name = model_name)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)

async def ask_all_stances(update: Update, context: ContextTypes.DEFAULT_TYPE):
    request = ' '.join(context.args)
    if request == '':
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No request")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Started...Usually takes 10-20 sec per each stance")
        # set params for summaries comparison
        n_tokens_out = 256
        full_reply = True

        # get summaries for all stances
        summary_list = []
        for stance in ['voenkor', 'tv', 'inet propaganda', 'moder', 'altern']:
            reply_text = ask_media(request, dates=dates, stance=[stance], model_name = model_name, tokens_out = n_tokens_out, full_reply = full_reply)
            summary_list.append(str([stance])+ "\n" + reply_text)
            # status update
            print(f"Summary for stance {stance} added.")
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Summary for stance {stance} added. Text: {reply_text}")

        summary_list = '\n\n'.join(summary_list)
        # compare summaries
        compare_reply = compare_stances(request, summary_list, model_name = model_name)
        print("Finished opeani summary of stances")
        print(compare_reply)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=compare_reply)


#=== UNKNOWN COMMAND ===#
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I don't understand that command. Type /start to see possible commands.")


if __name__ == '__main__':
    application = ApplicationBuilder().token(tg_token).build()
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    start_handler = CommandHandler('start', start)
    # caps_handler = CommandHandler('caps', caps)
    # caps_handler1 = CommandHandler('caps1', caps1)
    date_handler = CommandHandler(['set_date', 'set_dates'], set_date)
    show_dates_handler = CommandHandler(['show_dates', 'show_date'], show_dates)
    reset_dates_handler = CommandHandler(['reset_dates', 'reset_date'], reset_dates)
    model_handler = CommandHandler('set_model', set_model)
    show_model_handler = CommandHandler('show_model', show_model)
    info_handler = CommandHandler('info', info)
    ask_voenkor_handler = CommandHandler('ask_voenkor', ask_voenkor)
    ask_alter_handler = CommandHandler('ask_alter', ask_alter)
    ask_inet_prop_handler = CommandHandler('ask_inet_prop', ask_inet_prop)
    ask_moder_handler = CommandHandler('ask_moder', ask_moder)
    ask_tv_handler = CommandHandler('ask_tv', ask_tv)
    ask_all_stances_handler = CommandHandler(['ask_all_stances', 'ask_all'], ask_all_stances)
    unknown_handler = MessageHandler(filters.COMMAND, unknown)

    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    # application.add_handler(caps_handler)
    # application.add_handler(caps_handler1)
    application.add_handler(date_handler)
    application.add_handler(show_dates_handler)
    application.add_handler(reset_dates_handler)
    application.add_handler(info_handler)
    application.add_handler(ask_voenkor_handler)
    application.add_handler(ask_alter_handler)
    application.add_handler(ask_inet_prop_handler)
    application.add_handler(ask_moder_handler)
    application.add_handler(ask_tv_handler)
    application.add_handler(model_handler)
    application.add_handler(show_model_handler)
    application.add_handler(ask_all_stances_handler)
    application.add_handler(unknown_handler)

    application.run_polling()