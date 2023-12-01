import os
from io import StringIO
import openai
import json
from IPython.display import display, HTML
from dotenv import load_dotenv
from datetime import datetime

# get the OPENAI_API_KEY from the environment (docker-compose.yml)
openai.organization = os.environ.get( 'OPENAI_ORGANIZATION_ID' )
openai.api_key = os.environ.get( 'OPENAI_API_KEY' )

# 引数
file_id_train = 'file-xxxxxxxxxxxxxxxxxxx'     # 学習用データのファイルID "file-xxxxxxxxxxxx"
model         = 'davinci'                      # チューニングするモデル （davinciでやむなし）


# 実行
FineTune = openai.FineTune.create(training_file   = file_id_train,              # 学習用データのファイルID
                                  model           = model,                      # モデル
                                )
print(FineTune)

# 状況チェック
FineTune_data = FineTune.list().data
num           = len(FineTune_data)

for i in range(num):
    timestamp     = FineTune_data[i].created_at
    datetime      = datetime.fromtimestamp(timestamp)
    fine_tuned_id = FineTune_data[i].id
    status        = openai.FineTune.retrieve(id=fine_tuned_id).status
    model         = openai.FineTune.retrieve(id=fine_tuned_id).fine_tuned_model
    
    print(f'Create At: {datetime}')
    print(f'FineTune ID: {fine_tuned_id}')
    print(f'Model: {model}')
    print(f'Statsu: {status}\n')