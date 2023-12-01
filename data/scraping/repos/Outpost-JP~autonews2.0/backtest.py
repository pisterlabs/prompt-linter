#コンテンツ内容テスト

'''import asyncio
from langchain.document_loaders import HNLoader

# 非同期でHNLoaderを使用してデータをロードし、テキストファイルに保存する
async def main():
    # HNLoaderを初期化して最新のコンテンツをロード
    loader = HNLoader("https://news.ycombinator.com/newest")
    documents = loader.load()  # 非同期処理を待つ

    # テキストファイルに保存
    with open("hn_content.txt", "w", encoding='utf-8') as file:
        for doc in documents:
            # ドキュメントの内容をそのままテキストとして書き込む
            file.write(str(doc) + "\n\n")

# asyncioを使用してメイン関数を実行
asyncio.run(main())
'''

#base64デコードテスト
import os
import base64

GOOGLE_CREDENTIALS_BASE64 = os.getenv('CREDENTIALS_BASE64')
# Base64エンコードされたGoogleクレデンシャルをデコード
creds = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')
print(creds)
'''
'''
#base64エンコードテスト
'''
import base64

# JSONファイルのパスを指定
file_path = 'C:/Users/araki/Downloads/div/ニュース用クレデンシャル.json'

# ファイルを読み込み
with open(file_path, 'rb') as file:
    json_data = file.read()

# Base64エンコード
encoded_data = base64.b64encode(json_data)

# エンコードされたデータを文字列として出力
print(encoded_data.decode('utf-8'))


import os
from openai import OpenAI
'''
'''
content = 
    try:
        tools = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_content",
            "description": "Evaluate content based on various criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "importance": {
                        "type": "integer",
                        "description": "How impactful the topic of the article is. Scale: 0-10.",
                    },
                    "timeliness": {
                        "type": "integer",
                        "description": "How relevant the information is to current events or trends. Scale: 0-10.",
                    },
                    "objectivity": {
                        "type": "integer",
                        "description": "Whether the information is presented without bias or subjective opinion. Scale: 0-10.",
                    },
                    "originality": {
                        "type": "integer",
                        "description": "The novelty or uniqueness of the content. Scale: 0-10.",
                    },
                    "target_audience": {
                        "type": "integer",
                        "description": "How well the content is adjusted for a specific audience. Scale: 0-10.",
                    },
                    "diversity": {
                        "type": "integer",
                        "description": "Reflection of different perspectives or cultures. Scale: 0-10.",
                    },
                    "relation_to_advertising": {
                        "type": "integer",
                        "description": "If the content is biased due to advertising. Scale: 0-10.",
                    },
                    "security_issues": {
                        "type": "integer",
                        "description": "Potential for raising security concerns. Scale: 0-10.",
                    },
                    "social_responsibility": {
                        "type": "integer",
                        "description": "How socially responsible the content presentation is. Scale: 0-10.",
                    },
                    "social_significance": {
                        "type": "integer",
                        "description": "The social impact of the content. Scale: 0-10.",
                    }
                },
                "required": ["importance", "timeliness", "objectivity", "originality", "target_audience", "diversity", "relation_to_advertising", "security_issues", "social_responsibility", "social_significance"],
            },
        }
    }
]

async def(content)
    messages = [{
      "role": "system",
      "content": "あなたは優秀な先進技術メディアのキュレーターです。信頼性,最新性,重要性,革新性,影響力,関連性,包括性,教育的価値,時事性,倫理性をもとに、与えられた文章を10点満点でスコアリングして、JSON形式で返します。平均点は5点でスコアを付けるようにしてください。\n各基準は以下です。\n重要性 (Importance): 記事がどれだけ影響力のあるトピックに言及しているか。\n時宜性 (Timeliness): 情報が現在の出来事やトレンドにどれだけ適応しているか。\n客観性 (Objectivity): 情報がバイアスや主観的意見なしに提示されているか。\n独自性 (Originality): コンテンツが新規性や独創性を持っているか。\nターゲットオーディエンス (Target Audience): コンテンツが特定の聴衆に適切に調整されているか。\n多様性 (Diversity): 異なる視点や文化が反映されているか。\n広告との関連 (Relation to Advertising): コンテンツが広告によって偏っていないか。\nセキュリティ問題 (Security Issues): 情報がセキュリティ上の懸念を引き起こす可能性があるか。\n社会的責任 (Social Responsibility): コンテンツが社会的に責任ある方法で提示されているか。\n社会的重要性 (Social Significance): コンテンツが社会的な影響を持っているか。\n\nこれらの点数のみを出力してください。"
        },
        {
      "role": "user",
      "content": content
         }]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        )
    print(completion)
    except Exception as e:
    print(f"判別時にエラーが発生しました。: {e}")
    return ""
'''

paramater = '''
{
    "properties": {
        "importance": {
            "type": "integer",
            "description": "How impactful the topic of the article is. Scale: 0-10."
        },
        "timeliness": {
            "type": "integer",
            "description": "How relevant the information is to current events or trends. Scale: 0-10."
        },
        "objectivity": {
            "type": "integer",
            "description": "Whether the information is presented without bias or subjective opinion. Scale: 0-10."
        },
        "originality": {
            "type": "integer",
            "description": "The novelty or uniqueness of the content. Scale: 0-10."
        },
        "target_audience": {
            "type": "integer",
            "description": "How well the content is adjusted for a specific audience. Scale: 0-10."
        },
        "diversity": {
            "type": "integer",
            "description": "Reflection of different perspectives or cultures. Scale: 0-10."
        },
        "relation_to_advertising": {
            "type": "integer",
            "description": "If the content is biased due to advertising. Scale: 0-10."
        },
        "security_issues": {
            "type": "integer",
            "description": "Potential for raising security concerns. Scale: 0-10."
        },
        "social_responsibility": {
            "type": "integer",
            "description": "How socially responsible the content presentation is. Scale: 0-10."
        },
        "social_significance": {
            "type": "integer",
            "description": "The social impact of the content. Scale: 0-10."
        }
        "reason": {
        "type": "string",
        "description": "the basis for each numerical score. Output in 1-sentence Japanese with respect to all parameters"
        }
    },
    "required": ["importance", "timeliness", "objectivity", "originality", "target_audience", "diversity", "relation_to_advertising", "security_issues", "social_responsibility", "social_significance", "reason"]
}
'''

'''
import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


contents = "イーサン・エラスキーは、Amazon Photosが写真と動画からデータを収集していることについて警告し、それを無効にする方法を説明しています。Amazon Photosはプライム会員に無制限の写真保存と5GBの動画保存を提供しており、エラスキー氏はそれを利用しています。彼はAmazon Photosの利用規約を調べていたところ、写真と動画の認識データがサービスを利用できなくなるまで保持されることを発見しました。このため、自分の写真や動画が彼らのモデルの対象になる可能性があると指摘しています。そのため、写真や動画がトレーニングデータセットに含まれないようにするために、Amazon Photosの設定から画像認識をオフにするように説明しています。"
completion = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  temperature=0,
  response_format={ "type":"json_object" },
  messages=[
    {"role": "system", "content": f'あなたは優秀な先進技術メディアのキュレーターです。信頼性,最新性,重要性,革新性,影響力,関連性,包括性,教育的価値,時事性,倫理性をもとに、"""{contents}"""を10点満点でスコアリングして、JSON形式で返します。平均点は5点でスコアを付けるようにしてください。"""{paramater}"""のJSON形式で返してください。'},
    {"role": "user", "content": contents}
  ]
)

print(completion.choices[0].message.content)

'''

#　要約関数を書き出す。
def summarize_content(content):
    try:
                # テキストを分割
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=5000, chunk_overlap=0, separator="."
        )
        texts = text_splitter.split_text(content)

        # 分割されたテキストをドキュメントに変換
        docs = [Document(page_content=t) for t in texts]

        # 要約チェーンを実行
        result = refine_chain({"input_documents": docs}, return_only_outputs=True)

        # 要約されたテキストを結合して返す
        return result["output_text"]
    except Exception as e:
        logging.error(f"要約処理中にエラーが発生しました: {e}")
        traceback.print_exc()
        return ""
    

def summarize_content(content):
    try:
        # テキストを分割するためのスプリッターを設定
        text_splitter = CharacterTextSplitter(
            chunk_size=5000,  # 分割するチャンクのサイズ
            chunk_overlap=0,  # チャンク間のオーバーラップ
            separator="."     # 文章を分割するためのセパレータ
        )
        texts = text_splitter.split_text(content)

        # 分割されたテキストをドキュメントに変換
        docs = [Document(page_content=t) for t in texts]

        # 要約チェーンを実行
        result = refine_chain({"input_documents": docs}, return_only_outputs=True)

        # 要約されたテキストを結合して返す
        return result["output_text"]
    except Exception as e:
        logging.error(f"要約処理中にエラーが発生しました: {e}")
        traceback.print_exc()
        return ""
