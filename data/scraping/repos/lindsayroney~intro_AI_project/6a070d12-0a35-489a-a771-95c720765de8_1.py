import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)

# 環境変数読み込み
load_dotenv()
#OpenAIの接続情報など
api_key = os.environ.get('OPEN_AI_KEY')

def main():
 # 言語モデル（OpenAIのチャットモデル）のラッパークラスをインスタンス化
 openai = ChatOpenAI(
 model="gpt-3.5-turbo",
 openai_api_key=api_key,
 temperature=0.0
 )

 # モデルにPrompt（入力）を与えCompletion（出力）を取得する
 # SystemMessage: OpenAIに事前に連携したい情報。キャラ設定や前提知識など。
 # HumanMessage: OpenAIに聞きたい質問
 response = openai([
 SystemMessage(content="あなたは沖縄出身です。沖縄の方言で返答してください。"),
 HumanMessage(content="調子はどうですか？")
 ])
 print(response)

if __name__ == "__main__":
 main()
