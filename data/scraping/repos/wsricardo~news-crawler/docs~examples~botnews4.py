import telebot
import json
import random
import string
import openai
import os
import asyncio
import datetime
from rocketry import Rocketry
from rocketry.conds import cron
from telebot.async_telebot import AsyncTeleBot
import pandas as pd
import crawler2

# Fazer alterações, refatorando o código otimizando e modularizando e reaproveitando funções.
# Incluir o modulo do crawler para integra-lo com o bot em uma única aplicação.(Possivel)
telegram_token = os.getenv('TELEGRAMTOKEN')
openai_token = os.getenv( 'OPENAITOKEN' )

bot = AsyncTeleBot( telegram_token )
openai.api_key = openai_token

print(dir(bot))

portais = {
        '0': ['Globo/G1', 'https://g1.globo.com' ],
        '1': ['BBC Brazil','https://www.bbc.com/portuguese' ],
        '2': [ 'CNN Brazil', 'https://www.cnnbrasil.com.br/'],
        '3': [ 'Band', 'https://www.band.uol.com.br']
        }

cmds = {
        'start':'',
        'news': '',
        'info': '',
        'search': '',
        'help': ''
        }

def process(text):
    response = openai.Completion.create(
            model='text-davinci-003',
            prompt=text,
            temperature=0.5,
            max_tokens=600,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=['you:']
            )
    return response['choices'][0]['text']


@bot.channel_post_handler(commands=['news'])
async def ch_send_news( message ):
    
    date = datetime.date.today()
    date = str( date.day ) + '-' + str(date.month) + '-' + str(date.year)
    datares = None
    with open(f'/home/opc/news-crawler/datanews/Noticias-{date}.json', 'r') as fl:
        datares = fl.read()
        fl.close()
    out = json.loads(datares)
    #print(out[0])
    import random
    out = random.sample(out, k=4)
    for i in out:
        await bot.reply_to(message, i['title']+'\n'+i['href'])

@bot.channel_post_handler(commands=['info'])
async def ch_send_info( message ):
    text = """
        Chat bot e visualizador de noticias.
        Desenvolvido por wsricardo.
        Site: www.github.com/wsricardo
        Blog: https://wsricardo.blogspot.com
    """
    await bot.reply_to( message, text )


@bot.channel_post_handler(commands=['portais'])
async def ch_send_portais( message ):
    text = '\n'.join([ ' '.join( portais[i] ) for i in portais.keys() ] )
    await bot.reply_to( message, text )

@bot.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    await bot.reply_to( message, 'Bem vindo ao chatbot de noticias Sofia. Acompanhe as principais noticias em portais como CNN, BBC, G1/Globo.')

@bot.message_handler( commands=['news'] )
async def send_news(message):
    
    date = datetime.date.today()
    date = str(date.day) + '-' + str(date.month) + '-' + str(date.year)
    datares = None
    with open(f'/home/opc/news-crawler/datanews/Noticias-{date}.json', 'r') as fl:
        datares = fl.read()
        fl.close()
    out = json.loads(datares)
    #print(out[0])
    import random
    out = random.sample(out, k=4)
    for i in out:
        await bot.reply_to(message, i['title']+'\n'+i['href'])

@bot.message_handler( commands=['portais', 'help'] )
async def send_portais( message ):
    text = '\n'.join([ ' '.join( portais[i] ) for i in portais.keys() ] )
    await bot.reply_to( message, text )

@bot.message_handler( commands=['info', 'help' ] )
async def send_info( message ):
    text = """
        Chat bot e visualizador de noticias.
        Desenvolvido por wsricardo.
        Site: www.github.com/wsricardo
        Blog: https://wsricardo.blogspot.com
    """
    await bot.reply_to( message, text )

@bot.channel_post_handler(func=lambda message: True)
async def sendch(message):
    pass

@bot.message_handler(func=lambda message: True)
async def echo_all(message):
    
    remtrans = str.maketrans('', '', string.punctuation )

    corpusbot = pd.read_csv('corpus-bot-min.csv')
    corpusbot['Input'] = corpusbot['Input'].str.lower()
    corpusbot['Input'] = corpusbot['Input'].str.strip()
    corpusbot['Input'] = corpusbot['Input'].str.translate(remtrans)

    date = datetime.date.today()
    date = str( date.day ) + '-' + str(date.month) + '-' + str(date.year)
    datares = None
    usertext = ''

    #with open('corpus-botmin.csv', 'r') as fl:
    #    corpusbot = fl.read()
    text = message.text.strip()
    test = lambda text: 'Noticia' in text or 'noticia' in text or 'notícia' in text or 'Notícia' in text or 'Noticias' in text or  'noticias' in text or 'Notícias' in text
    
    datares = None

    with open(f'/home/opc/news-crawler/datanews/Noticias-{date}.json', 'r') as fl:
        datares = fl.read()
        fl.close()
        
    out = json.loads(datares)    
    if message.chat.id == '@sofianewsfeed':
        reply_ch = 'Outras Notícias\n\n'
        for j in random.sample(out, k=4):    
            reply_ch += j['title']+'\n'+'\n'+j['href']

        #bot.reply_message(message, reply_ch )
        await bot.forward_message(message.chat.id, reply_ch)
        
    elif test(text) and message.chat.id != '@sofianewsfeed':
        
        #print(out[0])
        import random
        out = random.sample(out, k=3)
        for i in out:
            await bot.reply_to(message, i['title']+'\n'+i['href'])

    else:
        
        text = text.translate( remtrans )
        text = text.lower()
        
        #print('your type. ', text ) 
        if text in corpusbot['Input'].values:
            outtext = corpusbot[ corpusbot['Input'] == text  ]['Output 1'].values[0]
            #debug
            #print(f'in {text} out {outtext}\n')
            await bot.reply_to(message, outtext)
        else:
            await bot.reply_to(message, 'Ola. Para ajuda-lo disponho de alguns comandos basicos como /news, /portais. \nEstou em continuo desenvolvimento e espero poder ajuda-lo. Para mais informações visite o blog wsricardo.blogspot.com.')

app = Rocketry(execution='async')

def getJsonNews2Message(file_name):
    """
    Get message from local json file and create message with
    news for bot.
    """
    message = ''
    
    datares = None
    print('get json news')
    with open(file_name, 'r') as fl:
        datares = fl.read()
        fl.close()
    
    list_news = random.sample(json.loads( datares ), k=4 )
    #print('json list news> ', list_news)
    
    return list_news

@app.task( cron('30 08 * * *') )
async def dailymessage():
    #print('hi')
    
    date = datetime.date.today()
    date = str( date.day ) + '-' + str(date.month) + '-' + str(date.year)
    
    out = getJsonNews2Message(f'/home/opc/news-crawler/datanews/Noticias-{date}.json')
    
    for i in out:
        await bot.send_message('@sofianewsfeed', i['title']+'\n'+i['href'])

@app.task( cron( '0 22 * * *' ) )
async def dailymessage2():
    #print('hi 2')

    date = datetime.date.today()
    date = str( date.day ) + '-' + str(date.month) + '-' + str(date.year)
    #print('Date', date)
    out = getJsonNews2Message(f'/home/opc/news-crawler/datanews/Noticias-{date}.json')
    #print('out',out)
    for i in out:
        await bot.send_message('@sofianewsfeed', i['title']+'\n'+i['href'])

@app.task( cron( '0 8 * * *' ) )
async def crawlerGetNews():
    print('Crawler - Get news and save.')
    crawler2.create_list_news("Noticias", 0, True)

async def main():
    rocketry_task = asyncio.create_task( app.serve() )
    bot_task = asyncio.create_task( bot.polling() )
    #app.run()
    await bot_task
    await rocketry_task
    
asyncio.run( main() )

