import logging
import os

if os.getenv('API_ENV') != 'production':
    from dotenv import load_dotenv

    load_dotenv()

from fastapi import FastAPI, Request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, FlexSendMessage
import openai
import uvicorn
import re
import urllib.parse
import ast


logging.basicConfig(level=os.getenv('LOG', 'WARNING'))
logger = logging.getLogger(__file__)

app = FastAPI()
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
openai.api_key = os.getenv('OPENAI_API_KEY')


def is_url_valid(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        # domain...
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


def delete_strings(s):
    # Step 1: Delete all contents from '#' to the next '&' character
    s = re.sub(r'#[^&]*', '', s)

    # Step 2: If '&openExternalBrowser=1' is not at the end, add it
    if '&openExternalBrowser=1' != s:
        s += '&openExternalBrowser=1'
    return s


def create_gcal_url(
        title='看到這個..請重生',
        date='20230524T180000/20230524T220000',
        location='那邊',
        description=''):
    base_url = "https://www.google.com/calendar/render?action=TEMPLATE"
    event_url = f"{base_url}&text={urllib.parse.quote(title)}&dates={date}&location={urllib.parse.quote(location)}&details={urllib.parse.quote(description)}"
    return event_url + "&openExternalBrowser=1"


def arrange_flex_message(gcal_url: str, action: dict) -> FlexSendMessage:
    return FlexSendMessage(alt_text='行事曆網址', contents={
        "type": "bubble",
        "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {
                        "type": "button",
                        "style": "link",
                        "height": "sm",
                        "action": {
                            "type": "uri",
                            "label": "WEBSITE",
                            "uri": gcal_url
                        }
                    },
                    action
                ],
            "flex": 0
        }
    })


@app.post("/webhooks/line")
async def callback(request: Request):
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode('utf-8')

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        return 'Invalid signature. Please check your channel access token/channel secret.'

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text = event.message.text
    action = {
        "type": "text",
        "text": "♻️ 文字太長請自行剪貼 ♻️",
        "size": "lg",
        "wrap": True
    }
    if len(text) < 300:
        action = {
            "type": "text",
            "text": "♻️點我重新產生♻️",
            "action": {
                "type": "message",
                "label": "action",
                "text": text
            },
            "size": "lg",
            "wrap": True
        }

    try:
        # Use OpenAI API to process the text
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"""
            Source 你會幫我把內容都轉換為 google calendar 的邀請網址。
            Message 我會給你任何格式的訊息，需要整理裡面的內容並對應上google calendar 的渲染方式，中文字需要編碼。
            Channel 將內容整理成標題、時間、地點、描述。範例: ['與同事聚餐', '20230627T230000/20230627T233000', '美麗華', '具體描述']，並且要能整理出對應標題、行事曆時間、地點，其餘內容整理完後放在描述裡面，現在是 2023 年。
            Receiver 連結google行事曆表單需要點選的民眾。
            Effect 最後透過陣列回傳。

            {text}
            """}])

        logger.info(response.choices[0].message)
        processed_text: str = response.choices[0].message.content
        gcal_list: list = ast.literal_eval(processed_text)
        title = gcal_list[0] or 'TBC'
        date = gcal_list[1] or 'TBC'
        location = gcal_list[2] or 'TBC'
        desc = gcal_list[3] or 'TBC'
        gcal_url: str = create_gcal_url(title, date, location, desc)
        logger.info(f'Google URL: {gcal_url}')
        logger.info(f"Is it url? {is_url_valid(gcal_url)}")
    except Exception as e:
        logger.warning('--------------------')
        logger.warning(e)
        logger.warning('--------------------')
    if is_url_valid(gcal_url):
        response: FlexSendMessage = arrange_flex_message(
            gcal_url=gcal_url, action=action)
    else:
        response: TextSendMessage = TextSendMessage(text=str(gcal_url))
    # Send the processed text back to the user through LINE Bot
    line_bot_api.reply_message(
        event.reply_token,
        [
            TextSendMessage(text="點選以下網址前，先確認時間地點："),
            TextSendMessage(
                text=f'標題: {title}\n時間: {date}\n地點: {location}\n描述: {desc}'),
            response
        ]
    )


if __name__ == "__main__":
    port = int(os.environ.get('PORT', default=8080))
    debug = True if os.environ.get(
        'API_ENV', default='develop') == 'develop' else False
    logging.info('Application will start...')
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
