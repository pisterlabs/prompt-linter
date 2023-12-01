# load api keys, libraries
tg_token = open('tg_api.txt', 'r').readline().strip()
api_key = open('API_key.txt', 'r').readline().strip()

import pandas as pd 
import numpy as np
import openai
import csv
import os
import time

openai.api_key = api_key

import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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

### load pickle df & filter
def get_filtered_df(request, dates=None, sources=None, stance=None):
    # load dataset
    df_filtered = pd.read_pickle('../TG news channels/filtered/df_war_ukr_filtered_ada_emb.pkl')
    full_len = df_filtered.shape[0]
    # filter by date, source, stance
    if dates:
        if len(dates) == 1:
            df_filtered = df_filtered[df_filtered['date'] >= dates[0]]
        elif len(dates) == 2:
            df_filtered = df_filtered[(df_filtered['date'] >= dates[0]) & (df_filtered['date'] <= dates[1])]
    if sources:
        df_filtered = df_filtered[df_filtered['source'].isin(sources)]
    if stance:
        df_filtered = df_filtered[df_filtered['stance'].isin(stance)]

    print(f"Number of messages in the filtered dataset: {df_filtered.shape[0]}, {df_filtered.shape[0]/full_len*100:.2f}% of the full dataset")

    return df_filtered
## find top relevant news
# embed request
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# get top relevant news
def get_top_openai(df, request=None, request_emb=None, model="text-embedding-ada-002", top_n=10):
    if request_emb is None and request is None:
        print('Error: no request')
        return
    if request_emb is None:
        request_emb = get_embedding(request)
    # calculate cosine similarity for all news vs request
    df['cos_sim'] = df['emb'].apply(lambda x: np.dot(x, request_emb)/(np.linalg.norm(x)*np.linalg.norm(request_emb)))
    top_sim_news = df.sort_values(by='cos_sim', ascending=False).head(top_n)
    # collect links & similarities
    top_sim_news['link'] = top_sim_news.apply(lambda x: "https://t.me/"+str(x.channel_name)+"/"+str(x.msg_id)+" - "+str(round(x.cos_sim,3)), axis=1)
    news_links = '\n'.join(top_sim_news.link.tolist())
    # collect news
    news4request = '\n'.join(top_sim_news.cleaned_message.tolist())
    return news4request, news_links

## Ask OpenAI
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

    # get news
    df_filtered = get_filtered_df(request, dates=dates, sources=sources, stance=stance)
    news4request, news_links = get_top_openai(df_filtered, request=request, model="text-embedding-ada-002", top_n=10)
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

#=== TECHNICAL functions ===#

async def start_ru(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=\
                                   "Чтобы отфильтровать новости с определенной дате, введите /set_date ГГГГ-ММ-ДД (если набрать 2 даты, то вторая дата - окончание периода).\n \
Чтобы проверить даты, введите /show_dates.\n \
Чтобы спросить у какой-то группы источников, введите /ask_voenkor, /ask_alter, /ask_moder, /ask_inet_prop или /ask_tv + ваш вопрос.\n \
Чтобы сравнить группы источников между собой, введите /ask_all_stances + ваш вопрос.\n\
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
To ask & compare all stances, type /ask_all_stances + your question.\n\
Note1: Database contains only news from on war in Ukraine from 2022-02-01 till 2023-01-31.\n\
Note2: Answers are based on sample of 10 news, so it can't make a full picture.\n\
Note3: Like all LLM it may sometimes hallucinate. But you can check the refered news following the links\n\
# Note4: Запросы лучше делать в виде вопросов (желательно конкретнее)), а не общих тем (так настроены промпты). Например, 'какова ситуация с мирными жителями в Украине?' вместо 'мирные жители в Украине'\
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
        full_reply = False

        # get summaries for all stances
        summary_list = []
        for stance in ['voenkor', 'altern', 'inet propaganda', 'moder', 'tv']:
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
    show_dates_handler = CommandHandler('show_dates', show_dates)
    model_handler = CommandHandler('set_model', set_model)
    show_model_handler = CommandHandler('show_model', show_model)
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