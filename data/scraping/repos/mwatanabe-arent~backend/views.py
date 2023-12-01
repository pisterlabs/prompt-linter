import openai
import requests
import os
import json
import random
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from django.shortcuts import render

# Create your views here.
from googleapiclient.discovery import build
from rest_framework.views import APIView
from rest_framework.response import Response

# .env ファイルをロードして環境変数へ反映
from dotenv import load_dotenv
load_dotenv()


class YouTubeDataAPIView(APIView):
    def get(self, request):
        video_data = self.get_trend_youtube()

        comment_count = video_data['statistics'].get(
            'commentCount', 'Not available')
        print(comment_count)
        if (0 < int(comment_count)):
            comment_long_text = self.get_comment_merged(video_data['id'])
            print(comment_long_text)
            comment_youyaku = self.generate_embedding(comment_long_text)
        else:
            comment_youyaku = "コメントはありません"

        res = self.message_response(f"""あなたはYoutubeの動画をたのしく紹介する人です。はじめに何の動画の紹介なのかを明示して、次のYoutubeの内容を動画のコメントと合わせて楽しく紹介してください。
                        Title: {video_data['snippet']['title']}
                        コメントの反応: {comment_youyaku} """)
#                        Description: {video_data['snippet']['description']}

        json_string = """
{
    "question" : ["質問文1","質問文2","質問文3"]
}
"""

        sform = f"""
            次のメッセージから質問文を３つ作成してください。
            データのフォーマットはjsonデータ形式で返してください
            フォーマット{json_string}

            メッセージ:{res}
            """
        print(sform)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"{sform}"}],
        )
        print("-----------質問文-----------")
        answer = response["choices"][0]["message"]["content"]
        print(answer)

        retdata = {
            "message": res,
            "question_json": answer
        }
        # 辞書をJSON形式の文字列に変換する
        # json_data = json.dumps(retdata,ensure_ascii=False)

        return Response(retdata)

    # 一つだけ返す

    def get_trend_youtube(self):

        # APIキーを設定します。
        DEVELOPER_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
        YOUTUBE_API_SERVICE_NAME = "youtube"
        YOUTUBE_API_VERSION = "v3"

        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                        developerKey=DEVELOPER_KEY)
        # トレンドビデオを検索します。
        search_response = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            chart='mostPopular',
            regionCode='JP',
            maxResults=10,  # 最大50件の結果を取得
        ).execute()

        # search_response.get("items", []) の結果が配列であると仮定しています
        items = search_response.get("items", [])

        # 配列が空でない場合にランダムに要素を取得します
        if items:
            random_item = random.choice(items)
            print(random_item)
            return random_item
        else:
            print("配列が空です")
            return None

    def message_response(self, message):
        template = \
            """
            あなたは人間と会話するAIです。
            過去の会話履歴はこちらを参照: {history}
            Human: {input}
            AI:
            """
        # プロンプトテンプレート
        prompt_template = PromptTemplate(
            input_variables=["history", "input"],  # 入力変数
            template=template,              # テンプレート
            validate_template=True,                  # 入力変数とテンプレートの検証有無
        )
        # ====================================================================================
        # LLM作成
        # ====================================================================================
        LLM = OpenAI(
            model_name="text-davinci-003",  # OpenAIモデル名
            temperature=0,                  # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
            n=1,                  # いくつの返答を生成するか
        )
        # ====================================================================================
        # メモリ作成
        # ====================================================================================
        # メモリオブジェクト
        memory = ConversationBufferWindowMemory(
            input_key=None,      # 入力キー該当の項目名
            output_key=None,      # 出力キー該当の項目名
            memory_key='history',  # メモリキー該当の項目名
            return_messages=True,      # メッセージ履歴をリスト形式での取得有無
            human_prefix='Human',   # ユーザーメッセージの接頭辞
            ai_prefix='AI',      # AIメッセージの接頭辞
        )
        # ====================================================================================
        # LLM Chain作成
        # ====================================================================================
        # LLM Chain
        chain = LLMChain(
            llm=LLM,             # LLMモデル
            prompt=prompt_template,  # プロンプトテンプレート
            verbose=True,            # プロンプトを表示するか否か
            memory=memory,          # メモリ
        )
        # ====================================================================================
        # モデル実行
        # ====================================================================================
        # 入力メッセージ
        # message = "Pythonとは何ですか？"
        # LLM Chain実行
        result = chain.predict(input=message)
        # ====================================================================================
        # 出力イメージ
        # ====================================================================================
        # 出力
        result = result.strip()
        print(result)

        return result

    def generate_question(self, message):

        return ""

    def get_comment_merged(self, video_id):
        URL = 'https://www.googleapis.com/youtube/v3/'

        def get_comments(url, params):
            response = requests.get(url, params=params)

            if response.status_code != 200:
                print("Error: API request failed.")
                print("Response: ", response.text)
                return None

            return response.json()

        def print_comments(no, video_id, parent_id=None, cno=None, next_page_token=None):

            result = ""
            params = {
                'key': os.getenv("GOOGLE_CLOUD_API_KEY"),
                'part': 'snippet',
                'videoId': video_id,
                'textFormat': 'plaintext',
                'maxResults': 10 if parent_id else 11,
                'pageToken': next_page_token,
                'order': 'relevance',
            }

            if parent_id:
                params['parentId'] = parent_id

            url = URL + 'comments' if parent_id else URL + 'commentThreads'
            resource = get_comments(url, params)

            if resource is None:
                return no, cno

            for comment_info in resource['items']:
                if parent_id:
                    # If parent_id is specified, it means it is a reply.
                    text = comment_info['snippet']['textDisplay']
                    like_cnt = comment_info['snippet']['likeCount']
                    user_name = comment_info['snippet']['authorDisplayName']
                    # print('{:0=4}-{:0=3}\t{}\t{}\t{}'.format(no, cno, text.replace('\r', '\n').replace('\n', ' '), like_cnt, user_name))
                    print('{}\t{}\t{}'.format(text.replace(
                        '\r', '\n').replace('\n', ' '), like_cnt, user_name))
                    cno += 1
                else:
                    # If parent_id is not specified, it means it is a top-level comment.
                    text = comment_info['snippet']['topLevelComment']['snippet']['textDisplay']
                    like_cnt = comment_info['snippet']['topLevelComment']['snippet']['likeCount']
                    reply_cnt = comment_info['snippet']['totalReplyCount']
                    user_name = comment_info['snippet']['topLevelComment']['snippet']['authorDisplayName']
                    parentId = comment_info['snippet']['topLevelComment']['id']
                    # print('{:0=4}\t{}\t{}\t{}\t{}'.format(no, text.replace('\r', '\n').replace('\n', ' '), like_cnt, user_name, reply_cnt))
                    # print('{}\t{}\t{}\t{}'.format( text.replace('\r', '\n').replace('\n', ' '), like_cnt, user_name, reply_cnt))
                    comment_msg = '{}'.format(
                        text.replace('\r', '\n').replace('\n', ' '))
                    print(comment_msg)
                    result += comment_msg + "\n"

                    if reply_cnt > 0:
                        cno = 1
                        # print_comments(no, video_id, parentId, cno)
                    no += 1
            return result

        no = 1
        merged_msg = print_comments(no, video_id)
        return merged_msg

    def generate_embedding(self, long_text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=100,
            chunk_overlap=0,
            length_function=len,
        )
        document_list = text_splitter.create_documents([long_text])
        # エンベッディングの初期化
        embeddings = OpenAIEmbeddings()
        # ベクターストアにドキュメントとエンベッディングを格納
        db = Chroma.from_documents(document_list, embeddings)
        retriever = db.as_retriever()
        llm = OpenAI(model_name="text-davinci-003",
                     temperature=0, max_tokens=1000)

        # チェーンを作り、それを使って質問に答える
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever)

        query = "コメントではどのような意見が多いですか？500文字以内で答えてください"
        answer = qa.run(query)
        return answer
