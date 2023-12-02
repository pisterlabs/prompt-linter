import json
from langchain import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage

class TetrisAssistant:
    @staticmethod
    def get_chat_messages():
        return [
            {
                "role": "system",
                "content":
"""
質問には100文字以内で回答する。
必要に応じて"function"を実行し、その応答を用いて、"user"の言語に合わせた言語で100文字以内で回答する。
"""
            },
            {
                "role": "user",
                "content":"テトリスのルールを教えて。"
            },
            {
                "role": "assistant",
                "content": "テトリスは、異なる形のブロックを落としてきて、行を埋めるゲームです。行を完全に埋めると、その行は消えます。ブロックが画面上部に積み上げられるとゲームオーバーです。"
            }
        ]

    @staticmethod
    def get_chat_functions():
        return [
            {
                "name": "search_tetris_index",
                "description": 
"""
テトリスをpythonで操作することを通してプログラミングを学ぶ時に用いる資料を検索する。
この資料は以下の内容を含む。
・テトリスのルール（点数,ルール,ボード情報,フィールド情報）
・プログラムによるテトリスの操作方法（実行コマンド,ミノの説明,コマンド、各種ファイル,アートの作り方）
・環境構築（AI,docker,Git,Windows(WSL,PowerShell),Linux,Mac）
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "SearchContent": {
                            "type": "string",
                            "description": "テトリス対戦に関する情報を検索 e.g.ミノを消した時の点数は？"
                        }
                    },
                    "required": ["SearchContent"]
                }
            }
        ]

class TetrisIndexSearch:
    def __init__(self):
        # indexの読み込み
        storage_context = StorageContext.from_defaults(persist_dir='./storage')
        self.index = load_index_from_storage(storage_context)

    def search_tetris_index(self, SearchContent):
        # クエリの実行
        query_engine = self.index.as_query_engine()
        response = query_engine.query(SearchContent)

        result = {
            "response": response.response
        }
        return json.dumps(result)
