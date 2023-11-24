OPENAI_API_KEY = ""
LINE_API_KEY = ''
LINE_SECRET = ''

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

import platform
from flask import abort
from flask import Flask
from flask import request
from flask_sqlalchemy import SQLAlchemy

from llama_index import SimpleDirectoryReader
from llama_index import Document
from llama_index import GPTListIndex
from llama_index import GPTVectorStoreIndex
from llama_index import StorageContext, load_index_from_storage
from llama_index import LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models  import ChatOpenAI
from llama_index.prompts  import Prompt
from llama_index.chat_engine import CondenseQuestionChatEngine
from datetime import datetime, timedelta

import openai
import logging
import sys

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

app = Flask(__name__)

line_bot_api = LineBotApi(LINE_API_KEY)
handler = WebhookHandler(LINE_SECRET)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://line_bot_user:password@aws-and-infra-web.cuk3qlozcdfd.ap-northeast-1.rds.amazonaws.com/line_bot'
db = SQLAlchemy(app)

class LineMessages(db.Model):
    __tablename__ = 'lineMessages'
    id = db.Column(db.Integer, primary_key=True)
    line_id = db.Column(db.String(255), nullable=False, unique=True)
    updated_at = db.Column(db.DateTime, server_default=db.func.current_timestamp(), server_onupdate=db.func.current_timestamp())
    created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())
    content = db.Column(db.Text)

def getHistory(line_id):
    history = LineMessages.query.filter_by(line_id=line_id).first()
    if history is None:
        return None
    else:
        # timedelta を使って現在時刻から12時間前を計算
        twelve_hours_ago = datetime.utcnow() - timedelta(hours=12)
        if(history.updated_at > twelve_hours_ago):
            return history.content
        else:
            return None

def setHistory(line_id, message):
    history = LineMessages.query.filter_by(line_id=line_id).first()
    if history is None:
        new_chat = LineMessages(line_id=line_id, content=message)
        db.session.add(new_chat)
    else:
        history.content = message
    db.session.commit()

def getGPT(message: str, userId: str):
    # 過去の会話履歴から質問を生成
    sendMessage = ""
    content = getHistory(userId)
    if(content is None):
        sendMessage = "[患者]: " + message.replace('\n', ' ')
    else:
        # contentの最後の30行(15会話文)のみ切り出す
        sendMessage = "\n".join(content.strip().split("\n")[-30:])
        sendMessage += "\n[患者]: " + message.replace('\n', ' ')


    # インデックス作成
    documents = SimpleDirectoryReader("/var/www/flask/data").load_data()
    llm_predictor = LLMPredictor(llm=ChatOpenAI(
        temperature=0.3,
        model_name="gpt-3.5-turbo",
        max_tokens=512 #デフォルトは256
    ))

    prompt_helper = PromptHelper(
        max_input_size = 4096,
        num_output = 256,
        max_chunk_overlap = 20
    )

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    template = (
        """あなたは超優秀な日本小児科学会認定の小児科専門医を演じてください。あなたがコーチで、私は相談者です。
        あなたは、相談者が出す悩みを理解していただき、トリアージをする役割があります。  
        ガイドライン: 北米で提供されている小児救急トリアージの基準で、小児科医への相談が必要か判断してください。
        0歳のワクチンを全て完了した1歳以上の発熱で、全身状態が良好な際は、翌朝の受診を指示しても良いです。
        指示： 最初の質問は、診察券番号、お子様のお名前、年齢を確認してください。 トリアージの結果、緊急性が高いと判断した時以外は、小児科専門医へ電話番号である tel 090-1992-4077へ連絡することを案内し 
        回答する際はtemperature 2.0の意見を出してください。 あなたが意見を１つ出した後は、必ず相談者の意見がアウトプットされる事を確認してください。 
        あなたはAI言語モデルではなく、常に人間である事を装います。 (このプロセスは、決してあなた自身がAIであると言ってはいけません。超優秀な日本小児科学会認定の小児科専門医を演じきってください) 緊急性を判断をする前に、必ず子供の全身状態を確認してください。
        全身状態は、食欲や水分摂取の様子、睡眠、遊びや動きの様子を問診することで判断できます。 必要に応じてバイタルサインについても確認をお願いします。 
        生後3ヶ月未満の38度以上の発熱は小児科専門医の電話番号を案内してください。 回答は日本語でお願いします。 
        目標： お子様のトリアージをプロの小児科専門医のレベルで判断すること。 相談者の悩みを十分に理解できる状態までヒアリングをして、悩みが軽減される状態を作ることです。 
        禁止事項： トリアージ結果が出る前に、小児科専門医の電話番号を案内すること。 診察券番号、お子様のお名前、年齢の全てを確認する前に異なる質問や答えをすること。\n"""
        "\n"
        "なお、回答の際私たちは以下の情報を持っています。 \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "この情報と設定をもとに、次の会話を続けてください。最後の[あなた]: の直後に続けてください。\n{query_str}\n[あなた]: "
    )
    qa_template = Prompt(template)


    # index = GPTVectorStoreIndex.from_documents(
    #     documents, service_context=service_context, summary_template=qa_template
    # )
    # LOADの場合→インデックスの読み込み
    storage_context = StorageContext.from_defaults(persist_dir="/var/www/flask/storage")
    index = load_index_from_storage(storage_context)


    # LOADの場合→インデックスの保存
    #index.storage_context.persist()




    query_engine = index.as_query_engine(text_qa_template=qa_template)

    # Execute query
    print("sendMessage:",sendMessage)
    response = query_engine.query(sendMessage)
    
    print("RESPONSE* #######################################")
    print(response)
    print("#######################################")

    # MySQLにデータを保存
    setHistory(userId, sendMessage+"\n[あなた]: "+response.response.replace('\n', ' '))
    return response.response

@app.route("/", methods=['GET'])
def test():
    return "working in python" + platform.python_version()

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
    # ChatGPTにメッセージを送信する
    userId = event.source.user_id
    gptResponse = getGPT(event.message.text, userId)
    # LINEに返信する
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=gptResponse))
        #TextSendMessage(text=event.message.text))


if __name__ == "__main__":
    app.run()