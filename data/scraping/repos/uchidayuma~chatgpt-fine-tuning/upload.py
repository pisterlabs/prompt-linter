import os
from io import StringIO
import openai
import json
from IPython.display import display, HTML
from dotenv import load_dotenv

# get the OPENAI_API_KEY from the environment (docker-compose.yml)
openai.organization = os.environ.get( 'OPENAI_ORGANIZATION_ID' )
openai.api_key = os.environ.get( 'OPENAI_API_KEY' )

# 学習用データのファイルパス
filepath_train = "./data/training.jsonl"

# ファイルアップロード（学習）
upload_file_train = openai.File.create(
                                      file=open(filepath_train, "rb"), # ファイル（JSONL）
                                      purpose='fine-tune',             # ファイルのアップロード目的
                                            )

# 出力
print(upload_file_train)
