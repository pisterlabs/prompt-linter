import asyncio
import time
from io import BytesIO
import os
import datetime
from datetime import date
import re
import random
import json
import requests
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()

import feedparser
from colour import Color
from PIL import Image, ImageOps
import scipy
import scipy.cluster
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import nekos
import googlemaps
from googletrans import Translator
import openai
import discord
from discord.ext import commands, tasks
from discord.ext.commands import CommandNotFound

from PTT_jokes import PttJokes
from bot_data import food_a, food_j, food_c, YamYABot_murmur

Google_Map_API_key = 'Google_Map_API_key'
Discord_token = 'Discord_token'
osu_API_key = 'osu_API_key'
openai.api_key = 'openai_api_key'

intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='', intents=intents, help_command=None)


# Google map推薦餐廳
def googlemaps_search_food(search_food, search_place):
    gmaps = googlemaps.Client(key=Google_Map_API_key)
    location_info = gmaps.geocode(search_place)
    location_lat = location_info[0].get('geometry').get('location').get('lat')
    location_lng = location_info[0].get('geometry').get('location').get('lng')
    search_place_r = gmaps.places_nearby(keyword=search_food, location=str(location_lat)+', '+str(location_lng), language='zh-TW', radius=1000)
    name_list = []
    place_id_list = []
    rating_list = []
    user_ratings_total_list = []
    for i in search_place_r.get('results'):
        name_list.append(i.get('name'))
        place_id_list.append(i.get('place_id'))
        rating_list.append(i.get('rating'))
        user_ratings_total_list.append(i.get('user_ratings_total'))
    df_result = pd.DataFrame({'name':name_list, 'place_id':place_id_list, 'rating':rating_list, 'user_ratings_total':user_ratings_total_list})
    try:
        df_result = df_result.loc[df_result.rating>4].sample()
    except:
        df_result = df_result.sample()
    return df_result.name.values[0], df_result.place_id.values[0], df_result.rating.values[0], df_result.user_ratings_total.values[0]

# 顏色判斷用
def get_rating_color(beatmap_rating):
    color_list = list(Color("#4FC0FF").range_to(Color("#4FFFD5"),6))+ \
                 list(Color("#4FFFD5").range_to(Color("#7CFF4F"),6))+ \
                 list(Color("#7CFF4F").range_to(Color("#F6F05C"),9))+ \
                 list(Color("#F6F05C").range_to(Color("#FF8068"),18))+ \
                 list(Color("#ff666b").range_to(Color("#FF3C71"),11))+ \
                 list(Color("#FF3C71").range_to(Color("#6563DE"),11))+ \
                 list(Color("#6563DE").range_to(Color("#2a27a1"),6))
    color_list = list(dict.fromkeys([str(i) for i in color_list]))
    rating_list = [1.5+(i/10) for i in range(61)]
    if beatmap_rating<1.5 :
        return('#4FC0FF', '1.5')
    for color, rating in zip(color_list, rating_list):
        if beatmap_rating>=rating and beatmap_rating<(rating+0.1):
            return(color, str(rating))
    if beatmap_rating>7.5 and beatmap_rating<8 :
        return('#2a27a1', '7.5')
    if beatmap_rating>=8 :
        return('#18158E', '8.0+')
    
# 天數換算用
def parse_date(td):
    resYear = float(td.days)/364.0
    resMonth = int((resYear - int(resYear))*364/30)
    resYear = int(resYear)
    resDay = int(td.days-(364*resYear+30*resMonth))
    return str(resYear) + " years " + str(resMonth) + " months and " + str(resDay) + " days."
    
# 取得AniList隨機角色
def get_AniList_character(AniList_userName, character_gender_input):
    query = '''
    query ($userName: String, $MediaListStatus: MediaListStatus, $page: Int, $perPage: Int) {
        Page (page: $page, perPage: $perPage) {
            pageInfo {hasNextPage}
            mediaList (userName: $userName, status: $MediaListStatus) {
                media {title{romaji}
                       characters{nodes{name{full native} gender image{medium}}}
                  }
            }
        }
    }
    '''
    page_number = 1
    next_page = True
    
    character_list = []
    character_image_list = []
    character_gender_list = []
    anine_title_list = []
    while next_page is True:
        variables = {'userName': AniList_userName, 'MediaListStatus': 'COMPLETED', 'page': page_number, 'perPage': 50 }
        response = requests.post('https://graphql.anilist.co', json={'query': query, 'variables': variables}).json()
        next_page = response.get('data').get('Page').get('pageInfo').get('hasNextPage')
        for anime in response.get('data').get('Page').get('mediaList'):
            characters = anime.get('media').get('characters').get('nodes')
            for character in characters:
                #character_name = character.get('name').get('full')
                character_native = character.get('name').get('native')
                character_image = character.get('image').get('medium')
                character_gender = character.get('gender')
                if (character_native!=None) and (character_image!=None) and (character_gender!=None):
                    character_list.append(character_native)
                    character_image_list.append(character_image)
                    character_gender_list.append(character_gender)
        page_number += 1
    df_all_character = pd.DataFrame({'character':character_list, 'image':character_image_list, 'gender':character_gender_list})
    df_character = df_all_character.drop_duplicates().loc[df_all_character.gender==character_gender_input].sample()
    character_name = df_character.character.values[0]
    character_image = df_character.image.values[0]
    
    return character_name, character_image
    
# 取得 zerochan 圖片
def get_ani_image(search_name):
    res = requests.get('https://www.zerochan.net/'+search_name, headers={"User-Agent":"Defined"}, verify=False)
    soup = BeautifulSoup(res.text,"html.parser")
    page_str = soup.find(class_="pagination").find('span').find(text=True)
    page = int(re.search('of ([0-9]*)',page_str).group(1))
    if page>10:
        page=10
    url = []
    res = requests.get('https://www.zerochan.net/'+search_name+'?p='+str(random.randint(1,page)), headers={"User-Agent":"Defined"}, verify=False)
    soup = BeautifulSoup(res.text,"html.parser")
    for ele in soup.find_all(id="content"):
        for i in ele.find_all('img'):
            url.append(i.get('src'))
    img_url = [i for i in url if i != 'https://static.zerochan.net/download.png' 
                             and i != 'https://s1.zerochan.net/small.png'
                             and i != 'https://s1.zerochan.net/medium.png']
    return random.choice(img_url)

#################################################################################################################################################

# [自動推播] 
@tasks.loop(seconds=60)
async def broadcast():
    # wysi
    utc8_time = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%H%M")
    if utc8_time == '0727' and random.randint(1,14) <= 3: # 時間且機率發生
        channel = bot.get_channel(842463449467453491) # 指定頻道
        await channel.send('早安ヽ(○´∀`)ﾉ')


# [自動更新狀態]
@tasks.loop(seconds=15)
async def activity_auto_change():
    status_w = discord.Status.online
    activity_w = discord.Activity(type=discord.ActivityType.playing, name=random.choice(YamYABot_murmur))
    await bot.change_presence(status= status_w, activity=activity_w)


# [啟動]
@bot.event
async def on_ready():
    print('目前登入身份：', bot.user)
    #broadcast.start() # 自動推播
    activity_auto_change.start() #自動更新狀態
    
    
# [新進成員] (依伺服器)
@bot.event
async def on_member_join(member):
    
    # 多樂一甲
    if member.guild.id == 885329184166137906:
        channel = bot.get_channel(893025355722539019)
        await channel.send(f"{member.mention} 進來後請把暱稱改成本名")
        
        
# [指令] YamYA_info : 取得呱YA所有所在伺服器列表
@bot.command()
async def YamYA_info(ctx):
    # 開發限定使用
    if int(ctx.message.author.id)==378936265657286659:
        guilds = bot.guilds
        all_server_list = [guild.name for guild in guilds]
        member_count_list = [guild.member_count for guild in guilds]
        server_owner_list = [bot.get_user(int(guild.owner_id)) for guild in guilds]

        all_server_count = len(all_server_list)
        all_member_count = sum(member_count_list)

        description_main = ''
        for server_name, member_number, owner in zip(all_server_list, member_count_list, server_owner_list):
            description_main = description_main+server_name+'\n--------'+str(member_number)+'人 from: '+f'{owner}'+'\n'
        # 卡片
        embed = discord.Embed(title='YamYA Bot Join Server Info', description=description_main)
        embed.set_footer(text='> 伺服器數量:'+str(all_server_count)+'  總人數:'+str(all_member_count))
        await ctx.send(embed=embed)
    
    
# [指令] 呱YA : 和呱YA聊天
'''
@bot.command(aliases=['gpt','GPT'])
async def 呱YA(ctx, *args):
    
    if len(args)==0:
        await ctx.send(random.choice(YamYABot_murmur))
        
    else :
        input_text = ' '.join(args)
        resp = [None]
        #def get():
        #    resp[0] = requests.post('https://asia-east2-bigdata-252110.cloudfunctions.net/ad_w2v_test',json={'input': input_text}).text
        def get():
            resp[0] = openai.Completion.create(engine="text-davinci-003",
                                                prompt=input_text,#.content,
                                                temperature=0.5,
                                                max_tokens=1024,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                               )["choices"][0]["text"]
        asyncio.get_event_loop().run_in_executor(None, get)
        while not resp[0]:
            await asyncio.sleep(0.5)
        await ctx.send(resp[0])
'''
        

# [指令] 代替呱YA說話
@bot.command()
async def 呱YA說(ctx, *, arg):
    #開發人員使用限定
    if int(ctx.message.author.id)==378936265657286659 or int(ctx.message.author.id)==86721800393740288:
        await ctx.message.delete()
        await ctx.send(arg)

    
# [指令] 笑話 :
@bot.command()
async def 笑話(ctx):
    ptt = PttJokes(1)
    joke_class_list = ['笑話','猜謎','耍冷','XD']
    error_n=0
    while True:
        try:
            joke_output = ptt.output()
            if joke_output[1:3] in joke_class_list and re.search('http',joke_output) is None:
                joke_output = re.sub('(\\n){4,}','\n\n\n',joke_output)

                joke_title = re.search('.*\\n',joke_output)[0]
                joke_foot = re.search('\\n.*From ptt',joke_output)[0]
                joke_main = joke_output.replace(joke_title,'').replace(joke_foot,'')
                break
        except:
            error_n+=1
            print(error_n)
            if error_n == 5:
                break
            pass
    embed = discord.Embed(title=joke_title, description=joke_main)
    embed.set_footer(text=joke_foot)
    await ctx.send(embed=embed)
    
    
# [指令] 新聞 :
@bot.command()
async def 新聞(ctx):
    d = feedparser.parse('https://news.google.com/rss?hl=zh-TW&gl=TW&ceid=TW:zh-Hant')
    n_title = [i.title for i in d.entries]
    source_name_list = [i.source.title for i in d.entries]
    title_list = [t.replace(' - '+s,'') for t,s in zip(n_title,source_name_list)] # 標題去除來源
    #published_list = [i.published for i in d.entries] #日期
    url_list = [i.link for i in d.entries]
    embed = discord.Embed(title=('頭條新聞'), description=(datetime.datetime.utcnow()+datetime.timedelta(hours=8)).strftime("%Y/%m/%d"), color=0x7e6487)
    for title, url, source in zip(title_list[:5], url_list[:5], source_name_list[:5] ):
        embed.add_field(name=title, value='['+source+']('+url+')', inline=False)
    news_message = await ctx.send('呱YA日報 '+(datetime.datetime.utcnow()+datetime.timedelta(hours=8)).strftime("%Y/%m/%d"), embed=embed)
    emojis = ['📰', '🎮', '🌤']
    for emoji in emojis:
        await news_message.add_reaction(emoji)
        
@bot.event
async def on_raw_reaction_add(payload):
    if payload.member.bot: # 機器人自身不算
        return
    channel = bot.get_channel(payload.channel_id)
    news_message = await channel.fetch_message(payload.message_id)    
    emoji = payload.emoji
    
    if news_message.content == '呱YA日報 '+(datetime.datetime.utcnow()+datetime.timedelta(hours=8)).strftime("%Y/%m/%d"): # 只對當日新聞指令有效
        
        if emoji.name == "📰":
            d = feedparser.parse('https://news.google.com/rss?hl=zh-TW&gl=TW&ceid=TW:zh-Hant')
            n_title = [i.title for i in d.entries]
            source_name_list = [i.source.title for i in d.entries]
            title_list = [t.replace(' - '+s,'') for t,s in zip(n_title,source_name_list)]
            url_list = [i.link for i in d.entries]
            google_embed = discord.Embed(title=('頭條新聞'), description=(datetime.datetime.utcnow()+datetime.timedelta(hours=8)).strftime("%Y/%m/%d"), color=0x598ad9)
            for title, url, source in zip(title_list[:5], url_list[:5], source_name_list[:5] ):
                google_embed.add_field(name=title, value='['+source+']('+url+')', inline=False)
            await news_message.edit(embed=google_embed)

        elif emoji.name == "🎮":
            d = feedparser.parse('https://gnn.gamer.com.tw/rss.xml')
            title_list = [i.title for i in d.entries]
            url_list = [i.link for i in d.entries]
            gnn_embed = discord.Embed(title=('巴哈姆特 GNN 新聞'), description=(datetime.datetime.utcnow()+datetime.timedelta(hours=8)).strftime("%Y/%m/%d"), color=0x598ad9)
            for title, url in zip(title_list[:5], url_list[:5]):
                gnn_embed.add_field(name=title, value='[巴哈姆特]('+url+')', inline=False)
            await news_message.edit(embed=gnn_embed)

        elif emoji.name == "🌤":
            # 取得台灣各縣市天氣
            url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/F-D0047-091?Authorization=rdec-key-123-45678-011121314'
            r = requests.get(url)
            data = r.json()['records']['locations'][0]['location']
            weather_embed = discord.Embed(title=('天氣預報 '), description=(datetime.datetime.utcnow()+datetime.timedelta(hours=8)).strftime("%Y/%m/%d"), color=0x598ad9)
            for loc_num, loc_name in zip([12,9,20,17,6], ['基隆','臺北','臺中','嘉義','臺南']):
                weather_data = data[loc_num]['weatherElement']
                rain = weather_data[0]['time'][0]['elementValue'][0]['value']
                temp = weather_data[1]['time'][0]['elementValue'][0]['value']
                weat = weather_data[6]['time'][0]['elementValue'][0]['value']
                weather_embed.add_field(name=loc_name ,value='☂'+rain+'%  🌡'+temp+'°C  ⛅'+weat, inline=False)
            # 香港天氣
            weat_hk = requests.get('https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=flw&lang=tc').json()['forecastDesc'].split("。")[1]
            forecast_hk = requests.get('https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=tc').json()['weatherForecast'][0]
            temp_hk = str(int((forecast_hk['forecastMaxtemp']['value']+forecast_hk['forecastMintemp']['value'])/2))
            rain_hk = str(int((forecast_hk['forecastMaxrh']['value']+forecast_hk['forecastMinrh']['value'])/2))
            weather_embed.add_field(name='香港' ,value='☂'+rain_hk+'%  🌡'+temp_hk+'°C  ⛅'+weat_hk, inline=False)
            await news_message.edit(embed=weather_embed)
            
            
# [指令] 地震 :
@bot.command()
async def 地震(ctx, *args):
    
    url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/E-A0015-001?Authorization=rdec-key-123-45678-011121314'
    eq_data = requests.get(url).json()
    eq_content = eq_data['records']['Earthquake'][0]['ReportContent']
    eq_image = eq_data['records']['Earthquake'][0]['ShakemapImageURI']
    ed_url = eq_data['records']['Earthquake'][0]['Web']
    
    embed=discord.Embed(title=eq_content, url=ed_url, color=0x636363)
    embed.set_image(url=eq_image)
    await ctx.send(embed=embed)


# [指令] 午/晚餐吃什麼:
@bot.command(aliases=['午餐吃什麼'])
async def 晚餐吃什麼(ctx, *args):
    ending_list = ['怎麼樣?','好吃',' 98','?','']
    # 沒有選類別的話就全部隨機: 吃土 2%  中式/台式 49%  日式/美式/意式 49%
    if len(args)==0:
        eat_dust = random.randint(1, 100)
        if eat_dust <= 2:
            await ctx.send('還是吃土?')
        else:
            eat_class = random.randint(1, 2)
            if eat_class == 1:
                await ctx.send(random.choice(food_c)+random.choice(ending_list))
            if eat_class == 2:
                await ctx.send(random.choice(food_j+food_a)+random.choice(ending_list))
    # 只輸入類別
    elif len(args)==1 and '式' in args[0]:
        food_class = args[0]
        if food_class=='中式' or food_class=='台式':
            await ctx.send(random.choice(food_c)+random.choice(ending_list))
        elif food_class=='日式' :
            await ctx.send(random.choice(food_j)+random.choice(ending_list))
        elif food_class=='美式' :
            await ctx.send(random.choice(food_a)+random.choice(ending_list))
        else:
            await ctx.send('我不知道'+food_class+'料理有哪些，請輸入中/台式、日式或美式 º﹃º')
    # 只輸入地點
    elif len(args)==1 and '式' not in args[0]:
        search_food = random.choice(food_j+food_a+food_c)
        search_place = args[0]
        try:
            restaurant = googlemaps_search_food(search_food, search_place)
            embed = discord.Embed(title=restaurant[0], description='⭐'+str(restaurant[2])+'  👄'+str(restaurant[3]), url='https://www.google.com/maps/search/?api=1&query='+search_food+'&query_place_id='+restaurant[1])
            embed.set_author(name = search_food+random.choice(ending_list))
            await ctx.send(embed=embed)
        except:
            await ctx.send('在'+search_place+'找不到適合的'+search_food+'餐廳，請再重新輸入一遍或換個地點名稱><')
    # 輸入類別和地點
    elif len(args)==2 and ('中式' in args[0] or '台式' in args[0] or '日式' in args[0] or '美式' in args[0]):
        food_class = args[0]
        search_place = args[1]
        if food_class=='中式' or food_class=='台式':
            search_food = random.choice(food_c)
        elif food_class=='日式' :
            search_food = random.choice(food_j)
        elif food_class=='美式' :
            search_food = random.choice(food_a)
        try:
            restaurant = googlemaps_search_food(search_food, search_place)
            embed = discord.Embed(title=restaurant[0], description='⭐'+str(restaurant[2])+'  👄'+str(restaurant[3]), url='https://www.google.com/maps/search/?api=1&query='+search_food+'&query_place_id='+restaurant[1])
            embed.set_author(name = search_food+random.choice(ending_list))
            await ctx.send(embed=embed)
        
        except:
            await ctx.send('在'+search_place+'找不到適合的'+search_food+'餐廳，請再重新輸入一遍或換個地點名稱><')
    # 格式打錯
    else:
        await ctx.send('確認一下指令是否正確: ```午餐吃什麼 [中式/台式/日式/美式] [地點]``` 參數皆可省略')


# [指令] 翻譯 :
@bot.command(aliases=['translate'])
async def 翻譯(ctx, *args):
    input_text = ' '.join(args)
    
    translator = Translator()
    us_trans = translator.translate(input_text, dest='en').text
    tw_trans = translator.translate(input_text, dest='zh-tw').text
    kr_trans = translator.translate(input_text, dest='ko').text
    jp_trans = translator.translate(input_text, dest='ja').text
    cn_trans = translator.translate(input_text, dest='zh-cn').text
    
    trans_list = [us_trans, tw_trans, kr_trans, jp_trans, cn_trans]
    output_text = ''
    for trans in trans_list:
        if input_text!=trans:
            output_text = output_text+trans+'\n'
            
    embed=discord.Embed(title='🌏 '+input_text, description=output_text, color=0x3884ff)
    await ctx.send(embed=embed)


# [指令] 全婆俠 :
@bot.command()
async def 全婆俠(ctx, *args):
    AniList_userName = ' '.join(args)
    character_gender_input = random.choice(['Female','Male'])
    random_character = get_AniList_character(AniList_userName, character_gender_input)
    if character_gender_input == 'Male':
        wifu_gender = '老公'
    else:
        wifu_gender = '婆'
    embed=discord.Embed(title=AniList_userName+': '+random_character[0]+'我'+wifu_gender, color=0x7875ff)
    embed.set_image(url=random_character[1])
    await ctx.send(embed=embed)
    
# [指令] waifu :
@bot.command()
async def waifu(ctx, *args):
    AniList_userName = ' '.join(args)
    character_gender_input = 'Female'
    random_character = get_AniList_character(AniList_userName, character_gender_input)
    embed=discord.Embed(title=AniList_userName+': '+random_character[0]+'我婆', color=0x7875ff)
    embed.set_image(url=random_character[1])
    await ctx.send(embed=embed)
    
# [指令] husbando :
@bot.command()
async def husbando(ctx, *args):
    AniList_userName = ' '.join(args)
    character_gender_input = 'Male'
    random_character = get_AniList_character(AniList_userName, character_gender_input)
    embed=discord.Embed(title=AniList_userName+': '+random_character[0]+'我老公', color=0x7875ff)
    embed.set_image(url=random_character[1])
    await ctx.send(embed=embed)
    
    
# [指令] AMQ : 隨機選一首動畫OP/ED撥放
'''
@bot.command(aliases=['amq'])
async def AMQ(ctx, *args):
    AniList_userName = ' '.join(args)
    query = ''
    query ($userName: String, $MediaListStatus: MediaListStatus, $page: Int, $perPage: Int) {
        Page (page: $page, perPage: $perPage) {
            pageInfo {hasNextPage}
            mediaList (userName: $userName, status: $MediaListStatus) {
                media {title{romaji english native}
                  }
            }
        }
    }
    ''
    # COMPLETED
    page_number = 1
    all_anime_list = []
    next_page = True
    while next_page is True:
        variables = {'userName': AniList_userName, 'MediaListStatus': 'COMPLETED', 'page': page_number, 'perPage': 50 }
        response = requests.post('https://graphql.anilist.co', json={'query': query, 'variables': variables}).json()
        next_page = response.get('data').get('Page').get('pageInfo').get('hasNextPage')
        page_number += 1

        anime_list = []
        for anime in response.get('data').get('Page').get('mediaList'):
            romaji_title = anime.get('media').get('title').get('romaji')
            english_title = anime.get('media').get('title').get('english')
            if romaji_title:
                anime_list.append([romaji_title,english_title])
        all_anime_list = all_anime_list+anime_list
    # WATCHING
    page_number = 1
    next_page = True
    while next_page is True:
        variables = {'userName': AniList_userName, 'MediaListStatus': 'CURRENT', 'page': page_number, 'perPage': 50 }
        response = requests.post('https://graphql.anilist.co', json={'query': query, 'variables': variables}).json()
        next_page = response.get('data').get('Page').get('pageInfo').get('hasNextPage')
        page_number += 1

        anime_list = []
        for anime in response.get('data').get('Page').get('mediaList'):
            romaji_title = anime.get('media').get('title').get('romaji')
            english_title = anime.get('media').get('title').get('english')
            if romaji_title:
                anime_list.append([romaji_title,english_title])
        all_anime_list = all_anime_list+anime_list
    # 隨機選一首
    while True:
        try:
            saerch_name = random.choice(all_anime_list)
            animethemes = requests.get('http://animethemes-api.herokuapp.com/api/v1/search/'+saerch_name[0]).json()
            if len(animethemes.get('anime'))==0:
                animethemes = requests.get('http://animethemes-api.herokuapp.com/api/v1/search/'+saerch_name[1]).json()
            ######## 
            # 柯南回歸用:
            if 'Meitantei Conan' in saerch_name[0]:
                animethemes = requests.get('http://animethemes-api.herokuapp.com/api/v1/search/Meitantei Conan').json()
                saerch_name = ['Meitantei Conan','Detective Conan']
            # Another排錯:
            if saerch_name[1] == 'Another':
                continue
            ########
            anime_num = random.randint(0,len(animethemes.get('anime'))-1)
            animetheme_num = random.randint(0,len(animethemes.get('anime')[anime_num].get('themes'))-1)
            theme_type = animethemes.get('anime')[anime_num].get('themes')[animetheme_num].get('type')
            theme_title = animethemes.get('anime')[anime_num].get('themes')[animetheme_num].get('title')
            theme_url = animethemes.get('anime')[anime_num].get('themes')[animetheme_num].get('mirrors')[0].get('mirror')
            await ctx.send('**'+saerch_name[1]+'** '+theme_type+':  '+theme_title+'\n'+theme_url)
            break
        except:
            pass
'''

# [指令] 神麻婆 : 神麻婆卡片
@bot.command(aliases=['mapper'])
async def 神麻婆(ctx, *args):
    try:
        mapper = ' '.join(args)
        get_beatmaps = requests.get('https://osu.ppy.sh/api/get_beatmaps?k='+osu_API_key+'&u='+mapper).json()
        beatmaps = {}
        num = 0
        for i in get_beatmaps:
            beatmaps[num] = i
            num = num+1
        df_beatmaps  = pd.DataFrame.from_dict(beatmaps, "index")
        if df_beatmaps.head(1).creator_id.values[0] == '0':
            await message.channel.send('我找不到這位神麻婆的圖;w;')
        else:
            df_beatmaps['status'] = df_beatmaps.approved.map({'1':'Rank','4':'Love'}).fillna('Unrank')
            df_beatmaps['genre_id'] = df_beatmaps.genre_id.map({'1':'Unspecified', '2':'Video Game', '3':'Anime', '4':'Rock', '5':'Pop',
                                                                '6':'Other', '7':'Novelty', '8':'Hip Hop', '9':'Electronic', '10':'Metal', 
                                                                '11':'Classical', '12':'Folk', '13':'Jazz'})
            df_beatmaps['language_id'] = df_beatmaps.language_id.map({'1':'Unspecified', '2':'English', '3':'Japanese', '4':'Chinese', '5':'Instrumental',
                                                                      '6':'Korean', '7':'FrenchItalian', '8':'German', '9':'Swedish', '10':'Spanish', 
                                                                      '11':'Polish', '12':'Russian', '14':'Other'})
            df_beatmaps['artist_unicode'] = df_beatmaps['artist_unicode'].fillna(df_beatmaps['artist']) # 將title和artist的unicode遺失值用英文補齊
            df_beatmaps['title_unicode'] = df_beatmaps['title_unicode'].fillna(df_beatmaps['title'])
            df_beatmaps['genre_id'] = df_beatmaps['genre_id'].fillna('Unspecified') # 類別、語言 補遺失值
            df_beatmaps['language_id'] = df_beatmaps['language_id'].fillna('Unspecified')
            
            df_beatmaps = df_beatmaps.astype({'beatmapset_id':'int64','favourite_count':'int64','playcount':'int64'}) # 欄位資料型態
            df_beatmaps['approved_date'] = pd.to_datetime(df_beatmaps['approved_date'], format='%Y-%m-%d %H:%M:%S')
            df_beatmaps['submit_date'] = pd.to_datetime(df_beatmaps['submit_date'], format='%Y-%m-%d %H:%M:%S')
            df_beatmaps['last_update'] = pd.to_datetime(df_beatmaps['last_update'], format='%Y-%m-%d %H:%M:%S')
            df_beatmaps = df_beatmaps.groupby('beatmapset_id').agg({'beatmap_id':'count', 'status':'min', 'genre_id':'min', 'language_id':'min',
                                                                    'title_unicode':'min', 'artist_unicode':'min',
                                                                    'approved_date':'min', 'submit_date':'min', 'last_update':'min', 
                                                                    'favourite_count':'min', 'playcount':'sum'}).reset_index(drop=False)
            mapper_id = beatmaps[0].get('creator_id')
            mapper_name = requests.get('https://osu.ppy.sh/api/get_user?k='+osu_API_key+'&u='+mapper_id).json()[0].get('username')
            # 年齡
            mapping_age = parse_date(datetime.datetime.now() - df_beatmaps.submit_date.min())
            # 做圖數量
            mapset_count = format( len(df_beatmaps),',')
            rank_mapset_count = format( len(df_beatmaps.loc[(df_beatmaps.status=='Rank')]),',')
            love_mapset_count = format( len(df_beatmaps.loc[(df_beatmaps.status=='Love')]),',')
            # 收藏、遊玩數
            favorites_count = format( df_beatmaps.favourite_count.sum(),',')
            platcount_count = format( df_beatmaps.playcount.sum(),',')
            # 最新的圖譜
            New_mapset_id  = str(df_beatmaps.sort_values(by='last_update', ascending=False).head(1).beatmapset_id.values[0])
            New_mapset_artist = df_beatmaps.sort_values(by='last_update', ascending=False).head(1).artist_unicode.values[0]
            New_mapset_title = df_beatmaps.sort_values(by='last_update', ascending=False).head(1).title_unicode.values[0]
            
            # 卡片
            embed = discord.Embed(title=mapper_name, url='https://osu.ppy.sh/users/'+mapper_id, color=0xff85bc)
            embed.set_thumbnail(url="https://s.ppy.sh/a/"+mapper_id)
            embed.add_field(name="Mapping Age ",value=mapping_age, inline=False)
            embed.add_field(name="Beatmap Count ",value='✍'+mapset_count+'  ✅'+rank_mapset_count+'  ❤'+love_mapset_count, inline=True)
            embed.add_field(name="Playcount & Favorites ",value='▶'+platcount_count+'  💖'+favorites_count, inline=True)
            embed.add_field(name="New Mapset!  "+New_mapset_artist+" - "+New_mapset_title, value='https://osu.ppy.sh/beatmapsets/'+New_mapset_id ,inline=False)
            embed.set_footer(text=date.today().strftime("%Y/%m/%d"))
            await ctx.send(embed=embed)
    except:
        await ctx.send('我找不到這位神麻婆的圖;w;')


# [指令] icon bbcode: 輸出圖譜新版 icon bbcode
@bot.command()
async def icon(ctx, *args):
    if args[0]=='bbcode':
        try:
            beatmap_url = args[1]
            beatmap_id = re.search(r'beatmapsets\/([0-9]*)', beatmap_url).group(1)
            beatmap_meta = requests.get('https://osu.ppy.sh/api/get_beatmaps?k='+osu_API_key+'&s='+beatmap_id).json()
            beatmap_difficulty_list = [meta.get('version') for meta in beatmap_meta]
            beatmap_rating_list = [float(meta.get('difficultyrating')) for meta in beatmap_meta]
            df_beatmap = pd.DataFrame([beatmap_difficulty_list,beatmap_rating_list]).T.rename(columns={0:'difficulty', 1:'rating'}).sort_values(by='rating', ascending=True)

            print_str = ''
            for index, row in df_beatmap.iterrows():
                diff_rating = round(row['rating'],1)
                diff_bbcode = '[img]https://raw.githubusercontent.com/Azuelle/osuStuff/master/diffs/gradient/difficon_std_'+get_rating_color(diff_rating)[1]+'%4016-gap.png[/img] [color='+get_rating_color(diff_rating)[0]+'][b]'+row['difficulty']+'[/b][/color]\n'
                if len(print_str+diff_bbcode)>1990:  # 輸出上限2000字
                    await ctx.send(print_str)
                    print_str = ''
                print_str = print_str+diff_bbcode
            await ctx.send(print_str)
        except:
            await ctx.send('我找不到這張圖;w;')


# [指令] combo color : 根據BG推薦 combo color
@bot.command()
async def combo(ctx, *args):
    if args[0]=='color':
        beatmap_url = args[1]
        beatmap_id = re.search(r'beatmapsets\/([0-9]*)', beatmap_url).group(1)
        color_num = 6
        
        img_url = 'https://b.ppy.sh/thumb/'+str(beatmap_id)+'l.jpg'
        im = Image.open(requests.get(img_url, stream=True).raw)
        #im = im.resize((150, 150))      # optional, to reduce time
        ar = np.asarray(im)
        shape = ar.shape
        ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar, color_num)
        
        recommend_combo_color = ''
        
        color_hex = '{:02x}{:02x}{:02x}'.format(int(round(codes[0][0])), int(round(codes[0][1])), int(round(codes[0][2])))
        sixteenIntegerHex = int(color_hex, 16)
        readableHex = int(hex(sixteenIntegerHex), 0)
        
        num_int = 1
        for i in codes:
            rgb_str = str((round(i[0]), round(i[1]), round(i[2])))
            recommend_combo_color = recommend_combo_color+str(num_int)+'. '+rgb_str+'\n'
            num_int+=1
            
        embed=discord.Embed(description=recommend_combo_color, color=readableHex)
        embed.set_author(name='Combo Color Recommend', icon_url='https://raster.shields.io/badge/--'+color_hex+'.png')
        embed.set_thumbnail(url=img_url)
        await ctx.send(embed=embed)


# [指令] BG色情守門員 : 檢查BG有沒有色色   
@bot.command(aliases=['bg'])
async def BG(ctx, beatmap_url):
    beatmap_id = re.search(r'beatmapsets\/([0-9]*)', beatmap_url).group(1)
    bg_url = 'https://b.ppy.sh/thumb/'+beatmap_id+'l.jpg'
    safe_detect_text = requests.post('https://asia-east2-bigdata-252110.cloudfunctions.net/ad_safe_detect_test',json={'input': bg_url}).text
    
    text_list = ['UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY']
    output_list = ['?_?','🙂Nice bg!','🙂Nice bg!','😊Nice bg!','😳我覺得有點色','🤩太色了,我要去找GMT檢舉']
    for match_text, output_text in zip(text_list, output_list):
        if safe_detect_text == match_text:
            embed=discord.Embed(title="BG色情守門員", description=output_text, color=0xff8a8a)
            embed.set_thumbnail(url=bg_url)
            await ctx.send(embed=embed)


# [指令] 彩蛋GIF
@bot.command(aliases=['cuddle'])
async def 貼貼(ctx):
    embed=discord.Embed(title="ლ(╹◡╹ლ)", color=0xd8d097)
    embed.set_image(url=nekos.img('cuddle'))
    await ctx.send(embed=embed)
@bot.command(aliases=['hug'])
async def 抱抱(ctx):
    embed=discord.Embed(title="(つ´ω`)つ", color=0xd8d097)
    embed.set_image(url=nekos.img('hug'))
    await ctx.send(embed=embed)
@bot.command(aliases=['kiss'])
async def 親親(ctx):
    embed=discord.Embed(title="( ˘ ³˘)♥", color=0xd8d097)
    embed.set_image(url=nekos.img('kiss'))
    await ctx.send(embed=embed)
@bot.command(aliases=['feed me','feed'])
async def 餵我(ctx):
    embed=discord.Embed(title="ψ(｀∇´)ψ", color=0xd8d097)
    embed.set_image(url=nekos.img('feed'))
    await ctx.send(embed=embed)
@bot.command(aliases=['nya'])
async def 喵(ctx):
    embed=discord.Embed(title="ο(=•ω＜=)ρ⌒☆", color=0xd8d097)
    embed.set_image(url=nekos.img('ngif'))
    await ctx.send(embed=embed)
@bot.command(aliases=['poke'])
async def 戳(ctx):
    embed=discord.Embed(title="戳~", color=0xd8d097)
    embed.set_image(url=nekos.img('poke'))
    await ctx.send(embed=embed)
@bot.command(aliases=['baka'])
async def 笨蛋(ctx):
    embed=discord.Embed(title="バカ~", color=0xd8d097)
    embed.set_image(url=nekos.img('baka'))
    await ctx.send(embed=embed)
@bot.command(aliases=['幹你娘','fuck'])
async def 幹(ctx):
    embed=discord.Embed(title="-`д´-/", color=0xd8d097)
    embed.set_image(url=nekos.img('slap'))
    await ctx.send(embed=embed)
    
    
# [指令] 小千 :
@bot.command(aliases=['千醬','Arashi','嵐千砂都'])
async def 小千(ctx):
    img_url = get_ani_image('Arashi+Chisato')
    embed=discord.Embed(title='Arashi Chisato', color=0xff6e90)
    embed.set_image(url=img_url)
    await ctx.send(embed=embed)
    
# [指令] 鯊鯊 :
@bot.command(aliases=['Gura','gura'])
async def 鯊鯊(ctx):
    img_url = get_ani_image('Gawr+Gura')
    embed=discord.Embed(title='🦐 Gawr Gura 🦐', color=0x5cb8ff)
    embed.set_image(url=img_url)
    await ctx.send(embed=embed)   
    
# [指令] 佩克拉 :
@bot.command(aliases=['Pekora','pekora','Peko'])
async def 佩克拉(ctx):
    img_url = get_ani_image('Usada+Pekora')
    embed=discord.Embed(title='👯 Usada Pekora 👯', color=0xffffff)
    embed.set_image(url=img_url)
    await ctx.send(embed=embed)


    
    
# [指令] HoneyWorks : 隨機一張HW的圖
'''
@bot.command(aliases=['HoneyWorks'])
async def honeyworks(ctx):
    hw_search_number = 0
    while True:
        hw_url = 'https://hanipre.miraheze.org'
        r = requests.get(hw_url+'/w/index.php?profile=images&search=File%3ASC')
        soup = BeautifulSoup(r.text, 'html.parser')
        img_soup = soup.find_all(class_="image")
        if len(img_soup)!=0:
            img_source = hw_url + img_soup[0].get('href')
            img_r = requests.get(img_source)
            try:
                img_title = re.split('File:SC (.*).png', BeautifulSoup(img_r.text, 'html.parser').findAll(class_="firstHeading mw-first-heading")[0].text)[1]
            except:
                #非SC
                img_title = re.split('File:(.*).png', BeautifulSoup(img_r.text, 'html.parser').findAll(class_="firstHeading mw-first-heading")[0].text)[1]

            
            img_url = 'https:'+BeautifulSoup(img_r.text, 'html.parser').findAll('img')[1]['src']
            break
        else:
            #重新查詢
            hw_search_number += 1
            if hw_search_number>3:
                break
            continue
    embed=discord.Embed(title=img_title, color=0xf025f4)
    embed.set_image(url=img_url)
    await ctx.send(embed=embed)
'''


@bot.command(aliases=['Halloween','halloween','HappyHalloween'])
async def 萬聖節快樂(ctx):
    # 特效圖
    mask = Image.open('halloween_mask.png')#.convert('RGB')
    # 大頭貼
    url = ctx.author.avatar_url
    data = requests.get(url)
    im = Image.open(BytesIO(data.content))
    # 組合
    output = ImageOps.fit(im, mask.size, centering=(0.5, 0.5))
    output = output.convert('RGB')
    output.paste(mask, (0, 0), mask)
    # 存為BytesIO
    image_binary = BytesIO() 
    output.save(image_binary, 'PNG')
    image_binary.seek(0)
    # 輸出
    await ctx.send('🎃 '+ctx.message.author.mention+' Happy Halloween!! 🎃')
    await ctx.send(file=discord.File(fp=image_binary, filename='image.png'))


# [NSFW指令] 色色
class_list_nsfw = ['waifu','neko', 'blowjob']
@commands.is_nsfw()
@bot.command(aliases=['hentai','エロ'])
async def 色色(ctx):
    random_nsfw_class = random.choice(class_list_nsfw)
    nsfw_res = requests.get('https://api.waifu.pics/nsfw/'+random_nsfw_class, headers={"User-Agent":"Defined"}, verify=False)
    nsfw_pic = json.loads(nsfw_res.text)['url']
    embed=discord.Embed(color=0xf1c40f)
    embed.set_image(url=nsfw_pic)
    await ctx.send(embed=embed)
    
    
# [指令] YamYA_invite : 邀請碼
@bot.command(aliases=['YamYA_invite'])
async def invite(ctx):
    embed=discord.Embed(title="喜歡認識osu麻婆、看動畫、亂道早安晚安的discord機器人", description="👾[GitHub](https://github.com/tommy9301122/YamYA_bot)   🍠[邀請連結](https://discord.com/api/oauth2/authorize?client_id=877426954888962068&permissions=0&scope=bot)", color=0xcc8b00)
    embed.set_author(name="呱YA一號", icon_url="https://cdn.discordapp.com/attachments/378910821234769942/854387552890519552/unknown.png")
    await ctx.send(embed=embed)
    
    
# [指令] help : 呱YA一號 指令與功能一覽
@bot.command(aliases=['YamYA_help'])
async def help(ctx):
    embed=discord.Embed(title="呱YA一號 指令與功能一覽", url="https://github.com/tommy9301122/YamYA_bot", color=0x5f6791)
    embed.add_field(name="🎮osu!", value="`神麻婆 [mapper's osu!帳號]` \n `icon bbcode [圖譜url]` \n `combo color [圖譜url]` \n `bg [圖譜url]`", inline=False)
    embed.add_field(name="📺二次元", value="`全婆俠/waifu/husbando [AniList帳號]` \n `amq [AniList帳號]`", inline=False)
    embed.add_field(name="🔞NSFW", value="`色色`", inline=False)
    embed.add_field(name="🍜其它", value="`午餐/晚餐吃什麼 [中式/台式/日式/美式] [地區]` \n `新聞` \n `地震` \n `翻譯 [想翻譯的文字]`", inline=False)
    embed.add_field(name="⛏機器人相關", value="`invite` \n `help`", inline=False)
    await ctx.send(embed=embed)


# [忽略error / NSFW警告] : 忽略所有前綴造成的指令錯誤、指令變數輸入錯誤、NSFW警告
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        return
    if isinstance(error, commands.errors.NSFWChannelRequired):
        embed=discord.Embed(title="🔞這個頻道不可以色色!!", color=0xe74c3c)
        embed.set_image(url='https://media.discordapp.net/attachments/848185934187855872/1046623635395313664/d2fc6feb-a48e-4ff6-8cd9-689a0cb43ff5.png')
        return await ctx.send(embed=embed)
    raise error
    

# on_message
@bot.event
async def on_message(message):
    if message.author == bot.user: #排除自己的訊息，避免陷入無限循環
        return
    
    # 早安、晚安、owo
    if message.content.lower() == 'gm':
        await message.channel.send('gm (｡･∀･)ﾉﾞ')
        
    if message.content.lower() == 'gn':
        await message.channel.send('gn (¦3[▓▓]')
        
    if message.content.lower() == "owo":
        await message.channel.send(f"owo, {message.author.name}")

    # 訊息中包含 azgod (不分大小寫)
    str_az = re.search(r'[a-zA-Z]{5}', message.content)
    if str_az:
        if str_az.group(0).lower() == 'azgod':
            k = random.randint(0, 1)
            if k == 0:
                await message.channel.send("https://i.imgur.com/PT5gInL.png")
            if k == 1:
                await message.channel.send("AzRaeL isn't so great? Are you kidding me? When was the last time you saw a player can make storyboard that has beautiful special effect, amazing lyrics and geometry. Mapping with amazing patterns, perfect hitsounds and satisfying flow? AzRaeL makes mapping to another level, and we will be blessed if we ever see a taiwanese with his mapping skill and passion for the game again. Amateurre breaks records. Sotarks breaks records. AzRaeL breaks the rules. You can keep your statistics, I prefer the AzGoD.")

    await bot.process_commands(message)
    
bot.run(Discord_token)
