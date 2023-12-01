from flask import Flask, request, abort

import openai
import os
from pyChatGPT import ChatGPT
from Scrape import scrape
from Postgresql import database


from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

app = Flask(__name__)

line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = str(event.message.text)
    if msg == "餵食電之助":
        profile = line_bot_api.get_profile(event.source.user_id)
        user_name = profile.display_name #使用者名稱
        uid = profile.user_id # 發訊者ID
        
        myDatabase = database(user_name, uid)
        v = myDatabase.add_food()
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"{user_name}成功餵食{v}次"))
    
    elif msg == "查看餵食排行榜": 
        profile = line_bot_api.get_profile(event.source.user_id)
        user_name = profile.display_name #使用者名稱
        uid = profile.user_id # 發訊者ID
        
        myDatabase = database(user_name, uid)
        v = myDatabase.select_top()
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=v))
            
    elif msg == "熱銷商品比價GO":       
        myScrape = scrape()
        output = myScrape.scrape()       
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=str(output)))
    
    elif msg == "最新新聞追追追":
        myScrape = scrape()
        news_list = myScrape.news()
        carousel_template_message = TemplateSendMessage(
             alt_text='最新新聞推薦',
             template=CarouselTemplate(
                 columns=[
                     CarouselColumn(
                         thumbnail_image_url=news_list[0]["img_url"],
                         title=news_list[0]["title"],#title
                         text=f'作者:{news_list[0]["role"]}',#作者
                         actions=[
                             URIAction(
                                 label='馬上查看',
                                 uri=news_list[0]["news_url"]#文章連結
                             )
                         ]
                     ),
                     CarouselColumn(
                         thumbnail_image_url=news_list[1]["img_url"],
                         title=news_list[1]["title"],
                         text=f'作者:{news_list[1]["role"]}',
                         actions=[
                             URIAction(
                                 label='馬上查看',
                                 uri=news_list[1]["news_url"]
                             )
                         ]
                     ),
                     CarouselColumn(
                         thumbnail_image_url=news_list[2]["img_url"],
                         title=news_list[2]["title"],
                         text=f'作者:{news_list[2]["role"]}',
                         actions=[
                             URIAction(
                                 label='馬上查看',
                                 uri=news_list[2]["news_url"]
                             )
                         ]
                     )
                 ]
             )
        )
        line_bot_api.reply_message(event.reply_token, carousel_template_message)
        
    else:          
        openai.api_key = os.getenv('SESSION_TOKEN')
        response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=msg,
                max_tokens=300,
                temperature=0.5
                )
        completed_text = response['choices'][0]['text']
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text = completed_text[2:]))
        
if __name__ == "__main__":
    app.run()
