import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

columns = ["質問内容", "回答"]
json_format = """
{
    "質問内容": "<質問内容そのもの>",
    "根拠": "<その回答をする理由を50字以内で述べてください>",
    "回答": "<"はい"か"いいえ"か"わかりません"か"場合による"の4択から選択してください。それ以外は返さないでください。>",
}
"""
def answer_datum(datum, questions, world):
    system_message = f'''
    このチャットではあなたはjson以外出力しないでください。いかなる場合もjson以外の会話をしてはいけません。
    
    あなたは次の{world}についてこの後の質問に答えてください。

    フォーマットは以下のjsonです
    """
    {json_format}
    """
        
    あなたが回答すべき対象の{world}は以下です。

    名前
    "{datum["name"]}"
    
    特徴
    """
    {datum["text"]}
    """

    それでは質問を行います。
    '''
    human_message = questions
    chat = ChatOpenAI(temperature=0, request_timeout=180)
    responses = []
    for question in questions:
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f'''
                        次の質問に以下のフォーマットのjsonを返してください。
                        """
                        {json_format}
                        """
                        質問:"{question}"
                         ''')
        ]
        response = chat(messages)
        tmp = response.content.split("{")[1].split("}")[0] # json以外を含んでしまうことが有る。。。
        response_json = json.loads("{"+tmp+"}")
        response_json["回答"] = response_json["回答"].strip("。")
        if not response_json["回答"] in {'いいえ', 'はい', 'わかりません', '場合による'}:
            print(response.content)
            messages.append(response)
            messages.append(HumanMessage(content=f'''
先の回答"{response_json["回答"]}"は正しくありません。
指定した、["はい","いいえ","わかりません","場合による"]の4択に含まれていません。

もう一度、正しく以下のフォーマットのjsonになるように回答を修正して先の質問に回答してください。
"""
{json_format}
"""
                                         '''))
            response = chat(messages)
            print(response.content)
            tmp = response.content.split("{")[1].split("}")[0] # json以外を含んでしまうことが有る。。。
            response_json = json.loads("{"+tmp+"}")
            print(response_json)
        response_json["回答"] = response_json["回答"].strip("。")
        assert response_json["回答"] in {'いいえ', 'はい', 'わかりません', '場合による'}
        responses.append(response_json)        
    return responses