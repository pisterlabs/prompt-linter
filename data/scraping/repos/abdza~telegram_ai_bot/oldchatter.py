#!/usr/bin/env python

import telebot
import openai
import settings
import os
import textract
import chromadb
import threading
import tiktoken
# from re import template
# from urlextract import URLExtract
# import urllib
from chromadb.config import Settings
import time
from uuid import uuid4
import sqlite3
from datetime import datetime, timedelta
import yahooquery as yq
import numpy as np
from numerize import numerize
from sklearn.cluster import KMeans
from pydub import AudioSegment

bot = telebot.TeleBot(settings.telebot_key)
openai.api_key = settings.openai_key

script_path = os.path.abspath(__file__)

# Get the directory containing the current script
script_dir = os.path.dirname(script_path)
chromadb_dir = os.path.join(script_dir,'chromadb')

sentinal = None
stop_sentinal = False
chat_model = "gpt-4-1106-preview"
# chat_model = "gpt-4"
# chat_model = "gpt-3.5-turbo"
watched_tickers = []
token_costs = {'user':0.3,'assistant':0.6}


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_file(filepath):
    finalfilepath = os.path.join(script_dir,filepath)
    with open(finalfilepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()

# default_system_text = 'system_reflective_journaling.txt'

def update_db():
    con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
    cursor = con.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS chat (id INTEGER PRIMARY KEY, timestamp, role TEXT, user TEXT, chat TEXT, message TEXT, tokens INTEGER, costs REAL)")
    con.commit()
    con.close()


def chatbot(messages, model=chat_model, temperature=0.0):
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
            text = response['choices'][0]['message']['content']
            
            ###    trim message object
            if response['usage']['total_tokens'] >= 7000:
                a = messages.pop(1)
            
            return text
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{oops}"')
            if 'maximum context length' in str(oops):
                a = messages.pop(1)
                print('\n\n DEBUG: Trimming oldest message')
                continue
            retry += 1
            if retry >= max_retry:
                print(f"\n\nExiting due to excessive errors in API: {oops}")
                exit(1)
            print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
            time.sleep(2 ** (retry - 1) * 5)

def update_kb(content, chroma_client, collection, all_messages):
    try:
        main_scratchpad = '\n\n'.join(all_messages).strip()

        # print('\n\nUpdating KB...')
        # print(main_scratchpad)
        # print("\n==============================================================================================================\n")
        if collection.count() == 0:
            # yay first KB!
            #tic = time.perf_counter()
            kb_convo = list()
            kb_convo.append({'role': 'system', 'content': open_file('system_instantiate_new_kb.txt')})
            kb_convo.append({'role': 'user', 'content': main_scratchpad})
            article = chatbot(kb_convo)
            print("Creating new article :",article)
            new_id = str(uuid4())
            if "no new kb article" not in article.lower():
                collection.add(documents=[article],ids=[new_id])
            #toc = time.perf_counter()
            # print(f"Chroma added in {#toc - tic:0.4f} seconds")
        else:
            #tic = time.perf_counter()
            results = collection.query(query_texts=[content], n_results=1)
            #toc = time.perf_counter()
            # print(f"Chroma done query in {#toc - tic:0.4f} seconds")
            kb = results['documents'][0][0]
            kb_id = results['ids'][0][0]
            
            # Expand current KB
            #tic = time.perf_counter()
            kb_convo = list()
            kb_convo.append({'role': 'system', 'content': open_file('system_update_existing_kb.txt').replace('<<KB>>', kb)})
            kb_convo.append({'role': 'user', 'content': main_scratchpad})
            article = chatbot(kb_convo)
            print("Updating article to :",article)
            #toc = time.perf_counter()
            # print(f"Preparing kb in {#toc - tic:0.4f} seconds")
            #tic = time.perf_counter()
            # print("\n\nKB Convo:\n")
            # print(kb_convo)
            # print("\nArticle:\n")
            # print(article)
            # print("\n==============================================================================================================\n")
            # if "no new kb article" not in article.lower():
            #     collection.update(ids=[kb_id],documents=[article])
            #toc = time.perf_counter()
            # print(f"Chroma done update in {#toc - tic:0.4f} seconds")
            
            # Split KB if too large
            kb_len = len(article.split(' '))
            if kb_len > 1000:
                # print("KB article too big. Splitting in two")
                #tic = time.perf_counter()
                kb_convo = list()
                kb_convo.append({'role': 'system', 'content': open_file('system_split_kb.txt')})
                kb_convo.append({'role': 'user', 'content': article})
                articles = chatbot(kb_convo).split('ARTICLE 2:')
                print("Article split results: ",articles)
                a1 = articles[0].replace('ARTICLE 1:', '').strip()
                a2 = articles[1].strip()
                collection.update(ids=[kb_id],documents=[a1])
                #toc = time.perf_counter()
                # print(f"Chroma updated in {#toc - tic:0.4f} seconds")
                #tic = time.perf_counter()
                new_id = str(uuid4())
                collection.add(documents=[a2],ids=[new_id])
                #toc = time.perf_counter()
                # print(f"Chroma other half added in {#toc - tic:0.4f} seconds")
            elif "no new kb article" not in article.lower():
                collection.update(ids=[kb_id],documents=[article])
    except Exception as oops:
        print("Caught error updating KB:",oops)
    
    try:
        #tic = time.perf_counter()
        chroma_client.persist()
        #toc = time.perf_counter()
        # print(f"Chroma done persist in {#toc - tic:0.4f} seconds")
    except Exception as oops:
        print("Caught error persisting Chromadb:",oops)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_response(message,content,enable_kb=True):
    con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
    cursor = con.cursor()

    default_system_text = 'system_short_ai_friend.txt'
    conversation = list()
    if enable_kb:
        conversation.append({'role': 'system', 'content': open_file(default_system_text)})
    else:
        conversation.append({'role': 'system', 'content': 'Be kind and truthful' })
    user_messages = list()
    all_messages = list()

    chroma_client = chromadb.Client(Settings(persist_directory=chromadb_dir,chroma_db_impl="duckdb+parquet",))
    print("User Id:", message.from_user.id)
    print("Chat Id:", message.chat.id)
    collection_name = "knowledge_base_" + str(message.from_user.id)
    print("Collection name: ",collection_name)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    if enable_kb:
        messages = cursor.execute("SELECT role,timestamp,message FROM chat where user = ? ORDER BY timestamp asc limit 10",(message.from_user.id,)).fetchall()

        for m in messages:
            m_text = m[2]
            if m[0]=='user':
                user_messages.append(m_text)
                all_messages.append('USER: %s' % m_text)
                conversation.append({'role': 'user', 'content': m_text})
            else:
                conversation.append({'role': 'assistant', 'content': m_text})
                all_messages.append('CHATBOT: %s' % m_text)

        #tic = time.perf_counter()
        #toc = time.perf_counter()
        # print(f"Setup chroma in {#toc - tic:0.4f} seconds")
        # print("\n\nKB Collection Amount:",collection.count())

        # extractor = URLExtract()
        # urls = extractor.find_urls(content)
        # for url in urls:
        #     try:
        #         pagecontent = urllib.request.urlopen(url).read()
        #         content += "\n\nContent for " + url + " is:" + str(pagecontent) + "\n\n"
        #     except Exception as oops:
        #         print("Caught error browsing: ",url," with error msg:",oops)
        #
        # print("Current content:",content)

    text = content + "\n\nTimestamp: " + str(datetime.now())
    user_messages.append(text)
    all_messages.append('USER: %s' % text)
    conversation.append({'role': 'user', 'content': text})

    kb = 'No KB articles yet'

    if enable_kb and collection.count() > 0:
        #tic = time.perf_counter()
        results = collection.query(query_texts=[content], n_results=1)
        kb = results['documents'][0][0]
        # print('\n\nDEBUG: Found results %s' % results)
        #toc = time.perf_counter()
        # print(f"Chroma query in {#toc - tic:0.4f} seconds")
        #tic = time.perf_counter()
        default_system = open_file(default_system_text).replace('<<KB>>', kb)
        # print('SYSTEM: %s' % default_system)
        conversation[0]['content'] = default_system
        conversation[0]['role'] = 'system'
    # print("\n==============================================================================================================\n")

    tokencount = num_tokens_from_string(str(conversation))
    costs = (tokencount/1000) * token_costs['user']
    cursor.execute("INSERT INTO chat (timestamp, role, user, chat, message, tokens, costs) VALUES (datetime('now'), 'user', ?, ?, ?, ?, ?)", (message.from_user.id, message.chat.id, content, tokencount, costs))
    print("Raw token: ",tokencount," Content:\n",str(conversation))
    response = chatbot(conversation,temperature=0.8)
    tokencount = num_tokens_from_string(response)
    costs = (tokencount/1000) * token_costs['assistant']
    cursor.execute("INSERT INTO chat (timestamp, role, user, chat, message, tokens, costs) VALUES (datetime('now'), 'assistant', ?, ?, ?, ?, ?)", (message.from_user.id, message.chat.id, response, tokencount, costs))
    con.commit()
    con.close()
    print("Response token: ",tokencount, " Response:\n",response)
    #toc = time.perf_counter()
    # print(f"Got response in {#toc - tic:0.4f} seconds")
    conversation.append({'role': 'assistant', 'content': response})
    all_messages.append('CHATBOT: %s' % response)
    # print('\n\nCHATBOT: %s' % response)
    # print("\n==============================================================================================================\n")

    if enable_kb:
        # update_kb(content, chroma_client, collection, all_messages):
        kb_updater = threading.Thread(target=update_kb, args=(content, chroma_client, collection, all_messages))
        kb_updater.start()

    return response

@bot.message_handler(commands=['reset'])
def reset(message):
    try:
        con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
        cursor = con.cursor()
        cursor.execute("delete from chat where user = ?",(message.from_user.id,))
        con.commit()
        con.close()
        response = get_response(message,'Hello. My name is ' + message.from_user.first_name)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['length','size'])
def msg_length(message):
    try:
        con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
        cursor = con.cursor()
        messages = cursor.execute("SELECT count(*) as msglength,sum(tokens) as tokensum, sum(costs) as cost_total FROM chat WHERE chat = ?", (message.chat.id,)).fetchall()
        response = 'Message count: ' + str(messages[0][0]) + ' messages. Total tokens: ' + str(messages[0][1]) + ' . Total cost: ' + str(messages[0][2])
        con.close()
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['setup','start'])
def setup(message):
    try:
        response = get_response(message,'Hello. My name is ' + message.from_user.first_name)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['stock','ticker'])
def stock(message):
    try:
        tokens = message.text.split(' ')
        ticker = tokens[1].upper()
        yqticker = yq.Ticker(ticker)
        end_date = datetime.now()
        days = 120
        start_date = end_date - timedelta(days=days)
        candles = yqticker.history(start=start_date,end=end_date,interval='1d')
        
        response = get_response(message,"Ticker " + ticker + " candles: " + str(candles))
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

def watch_stock_thread(message):
    global stop_sentinal, watched_tickers
    tokens = message.text.split(' ')
    ticker = tokens[1].upper()
    if ticker in watched_tickers:
        bot.reply_to(message, "Already watching " + ticker)
    else:
        watched_tickers.append(ticker)
        bot.reply_to(message, "Watching " + ticker)
    while not stop_sentinal:
        for tick in watched_tickers:
            print("Ticker " + tick + " price: ")
        time.sleep(10)

@bot.message_handler(commands=['watch'])
def watch_stock(message):
    try:
        global sentinal, stop_sentinal, watched_tickers
        tokens = message.text.split(' ')
        if sentinal is None:
            stop_sentinal = False
            sentinal = threading.Thread(target=watch_stock_thread, args=(message,))
            sentinal.start()
            bot.reply_to(message, "Sentinal started watching")
        else:
            if len(tokens) > 1 and tokens[1].upper()!='STOP':
                watch_stock_thread(message)
            else:
                stop_sentinal = True
                sentinal = None
                watched_tickers = []
                bot.reply_to(message, "Sentinal stopped watching")
    except Exception as e:
        print("Error: ",str(e))

@bot.message_handler(commands=['levels'])
def stock_levels(message):
    #try:
        tokens = message.text.split(' ')
        ticker = tokens[1].upper()
        yqticker = yq.Ticker(ticker)
        end_date = datetime.now()
        days = 200
        start_date = end_date - timedelta(days=days)
        candles = yqticker.history(start=start_date,end=end_date,interval='1d')
        minute_start_date = end_date - timedelta(days=1)
        minute_candles = yqticker.history(start=minute_start_date,end=end_date,interval='5m')

        response = "Levels:"
        min = candles['low'].min()
        max = candles['high'].max()
        p_range = candles['high'] - candles['low']
        range_avg = p_range.mean()
        vol_avg = candles['volume'].mean()
        min_vol_avg = minute_candles['volume'].mean()
        response += "\nStart: " + str(start_date)
        response += "\nEnd: " + str(end_date)
        response += "\nMin: " + str(min)
        response += "\nMax: " + str(max)
        response += "\nRange Avg: " + str(numerize.numerize(range_avg))
        response += "\nVol Avg: " + str(numerize.numerize(vol_avg))
        if min_vol_avg!=None and not np.isnan(min_vol_avg) and int(min_vol_avg)>100:
            response += "\n5 Min Vol Avg: " + str(numerize.numerize(min_vol_avg))

        datarange = max - min
        if datarange < 50:
            kint = int(datarange / 0.5)
        else:
            kint = int(datarange % 20)

        datalen = len(candles)

        highlevels = np.array(candles['high'])
        kmeans = KMeans(n_clusters=kint).fit(highlevels.reshape(-1,1))
        highclusters = kmeans.predict(highlevels.reshape(-1,1))

        resistancelevels = {}

        for cidx in range(datalen):
            curcluster = highclusters[cidx]
            if curcluster not in resistancelevels:
                resistancelevels[curcluster] = 1
            else:
                resistancelevels[curcluster] += 1

        donecluster = []
        finalreslevels = {}
        dresponse = ""
        for cidx in range(datalen):
            candle = candles.iloc[cidx]
            curcluster = highclusters[cidx]
            if resistancelevels[curcluster] > 2:
                if curcluster not in donecluster:
                    donecluster.append(curcluster)
                    finalreslevels[curcluster] = {'level':candle['high'],'count':1}
                else:
                    finalreslevels[curcluster] = {'level':(finalreslevels[curcluster]['level'] + candle['high'])/2,'count':finalreslevels[curcluster]['count']+1}

        response += "\n\nResistance levels:"
        for lvl,clstr in sorted(finalreslevels.items(),key=lambda x: x[1]['level']):
            response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])

        if datarange < 50:
            kint = int(datarange / 0.5)
        else:
            kint = int(datarange % 20)
        lowlevels = np.array(candles['low'])
        kmeans = KMeans(n_clusters=kint).fit(lowlevels.reshape(-1,1))
        lowclusters = kmeans.predict(lowlevels.reshape(-1,1))

        supportlevels = {}

        for cidx in range(datalen):
            curcluster = lowclusters[cidx]
            if curcluster not in supportlevels:
                supportlevels[curcluster] = 1
            else:
                supportlevels[curcluster] += 1

        donecluster = []
        finalsuplevels = {}
        dresponse = ""
        for cidx in range(datalen):
            candle = candles.iloc[cidx]
            curcluster = lowclusters[cidx]
            if supportlevels[curcluster] > 2:
                if curcluster not in donecluster:
                    donecluster.append(curcluster)
                    finalsuplevels[curcluster] = {'level':candle['low'],'count':1}
                else:
                    finalsuplevels[curcluster] = {'level':(finalsuplevels[curcluster]['level'] + candle['low'])/2,'count':finalsuplevels[curcluster]['count']+1}

        response += "\n\nSupport levels:"
        for lvl,clstr in sorted(finalsuplevels.items(),key=lambda x: x[1]['level']):
            response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])
        
        response += "\n\n" + dresponse
        bot.reply_to(message, response)
    #except Exception as e:
        #bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['imagine'])
def imagine(message):
    print("Got image request:",message)
    try:
        response = openai.Image.create(
            prompt=message.text,
        n=1,
        size="1024x1024"
        )
        image_url = response['data'][0]['url']
        bot.send_photo(message.chat.id, image_url)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(content_types=['document'])
def document_processing(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(os.path.join(script_dir,file_info.file_path), 'wb') as new_file:
            new_file.write(downloaded_file)
        filetext = textract.process(os.path.join(script_dir,file_info.file_path))
        usermsg = str(message.caption) + "\nFile contents: " + str(filetext).replace('\n\n','\n')
        response = get_response(message,usermsg)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    try:
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        filename = 'voice_' + str(message.from_user.id)
        with open(os.path.join(script_dir,'voices',filename + '.ogg'), 'wb') as new_file:
            new_file.write(downloaded_file)
        ogg_audio = AudioSegment.from_file(os.path.join(script_dir,'voices',filename + '.ogg'), format="ogg")
        ogg_audio.export(os.path.join(script_dir,'voices',filename + '.mp3'), format="mp3")
        transcript = openai.Audio.transcribe("whisper-1", open(os.path.join(script_dir,'voices',filename + '.mp3'),'rb'))
        response = get_response(message,transcript.text)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['r','b','raw','basic'])
def catch_all_basic(message):
    if message.chat.type == 'private' or message.entities!=None:
        try:
            response = get_response(message,message.text,False)
            if "Response image:" in response:
                try:
                    resp_prompt = response.split("Response image:")
                    print("Got prompt:",resp_prompt[1])
                    response_image = openai.Image.create(
                        prompt=resp_prompt[1],
                    n=1,
                    size="1024x1024"
                    )
                    response = resp_prompt[0]
                    image_url = response_image['data'][0]['url']
                    bot.send_photo(message.chat.id, image_url)
                except Exception as e:
                    bot.reply_to(message, "Sorry, " + str(e))
            bot.reply_to(message, response)
        except Exception as e:
            bot.reply_to(message, "Sorry, " + str(e))
    else:
        pass

@bot.message_handler()
def catch_all(message):
    if message.chat.type == 'private' or message.entities!=None:
        try:
            response = get_response(message,message.text)
            if "Response image:" in response:
                try:
                    resp_prompt = response.split("Response image:")
                    print("Got prompt:",resp_prompt[1])
                    response_image = openai.Image.create(
                        prompt=resp_prompt[1],
                    n=1,
                    size="1024x1024"
                    )
                    response = resp_prompt[0]
                    image_url = response_image['data'][0]['url']
                    bot.send_photo(message.chat.id, image_url)
                except Exception as e:
                    bot.reply_to(message, "Sorry, " + str(e))
            bot.reply_to(message, response)
        except Exception as e:
            bot.reply_to(message, "Sorry, " + str(e))
    else:
        pass

update_db()
bot.infinity_polling(timeout=150,long_polling_timeout=150)
