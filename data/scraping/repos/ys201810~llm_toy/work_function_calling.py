# coding=utf-8
# 参考:　https://toukei-lab.com/function-calling
import sys
import openai
import pathlib
import json
base_path = pathlib.Path.cwd().parent
sys.path.append(str(base_path))
from utils import get_config
config_file = base_path / 'config' / 'config.yaml'
config = get_config.run(config_file)
openai.api_key = config.openai_api_key


def get_schedule(date, person):
    """
    スケジュール取得用の関数を定義
    - ここではdateとpersonがパラメータとして渡りそれに対するscheduleが返ってくる仕様
    - 本来であればパラメータを元に適切なスケジュールが返ってくるようになるべきだが簡単のために今回は固定値を返す
    """
    schedule_info = {
        "date": date,
        "person": person,
        "schedule": "10時：A社とMTG、12時：友人Bとランチ"
    }
    return json.dumps(schedule_info)


def get_test_score(test_kind, person):
    test_score_info = {
        "test_kind": test_kind,
        "person": person,
        "test_score": "国語は60点、算数は85点、社会は50点、理科は90点、英語は30点です。"
    }
    return json.dumps(test_score_info)

def run_conversation(prompt):
    # GPTのAPIへのリクエストを定義
    messages = [{"role": "user", "content": prompt}]
    functions = [
        {
            "name": "get_schedule",
            "description": "特定の日付のスケジュールを取得して返す",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "日付"
                    },
                    "person": {
                        "type": "string",
                        "description": "人の名前"
                    },
                },
                "required": ["date","person"]
            }
        },
        {
            "name": "get_test_score",
            "description": "国語・算数・社会・理科・英語のテストの成績を取得して返す",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_kind": {
                        "type": "string",
                        "description": "科目"
                    },
                    "person": {
                        "type": "string",
                        "description": "人の名前"
                    },
                },
                "required": ["test_kind", "person"]
            }
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto: 独自関数を利用するかどうかをGPTが自動で選択。
    )
    response_message = response["choices"][0]["message"]

    # response_message = json.loads(json.dumps(response_message))
    print(response)

    # 独自定義したfunctionを呼ぶかどうかの処理
    if response_message.get("function_call"):
        available_functions = {
            "get_schedule": get_schedule,
            "get_test_score": get_test_score,
        }
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        if function_name == 'get_schedule':
            function_response = function_to_call(
                date=function_args.get("date"),
                person=function_args.get("person")
            )
        elif function_name == 'get_test_score':
            function_response = function_to_call(
                test_kind=function_args.get("test_kind"),
                person=function_args.get("person")
            )

        # 独自関数のレスポンスを渡す
        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        # 独自関数のレスポンスをもとに改めてAPIにリクエスト
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response

    else:
        return response_message


def main():
    question = '田中さんの英語のテストの点数を教えて' # '田中さんの6/30のスケジュールを教えて' # 'chatGPTでできることを教えてください'
    result = run_conversation(question)
    if result.get('choices', '') == '':
        print(result['content'])
    else:
        print(result["choices"][0]["message"]["content"])



if __name__ == '__main__':
    main()
