from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
import os
import openai
from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
import logging

openai.api_key = os.environ["OPENAI_API_KEY"]
logger = logging.getLogger("DatasetGenerator")
logger.setLevel(logging.INFO)

prompt = """Pythonで1行のコードを生成し、StackOverflowの日本語の質問を解決してください。コメントや式は含めないでください。インポート文も不要です。

このタスクでは、入力は日本語のテキストで、変数名や操作が記述されています。出力は、そのタスクを達成するためのPythonの1行のコードです。コメントや式は含めないでください。インポート文も不要です。

input="スペースで区切られた入力`stdin`を変数に格納して表示する"
output="for line in stdin: a = line.rstrip().split(' ') print(a)"

input="リスト`word_list'内に出現する単語を数える"
output="Counter(word_list)"

input="tweepyインスタンス`api`を使い、文字列`word`を含んだツイートを検索し、結果をリストとして得る"
output="search = api.search(q=word)"

input="データベースの設定を表示する"
output="print(settings.DATABASES)"

input="ネストされているリスト`li`を見やすく表示する"
output="pprint.pprint(li)"

input="HTMLファイル'test.html'を開き、テキストオブジェクト'text'をutf-8で保存する"
output="f = open('test.html', 'w') f.write(text.encode('utf-8'))"
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
unlimited_dataset_generator = OpenAIDatasetGenerator(initial_temperature=0.5, max_temperature=1.5, responses_per_request=2, batch_size=3)
generated_dataset = unlimited_dataset_generator.generate_dataset_split(
    prompt_spec, 50, split=DatasetSplit.TRAIN
)
generated_dataset.save_to_disk("generated_dataset/jp2python")
