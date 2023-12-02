import openai
import streamlit as st
import json

openai.api_base = "https://oai.langcore.org/v1"

def function_calling(messages, functions, function_name):
    function_call = "auto"
    if function_name: 
        function_call = {"name": function_name}
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call=function_call
    )

    assert "choices" in response, response
    res = response["choices"][0]["message"] # type: ignore
    if "function_call" in res:
        return res["function_call"]["arguments"], True
    else :
        return res["content"], False


def main():
    st.title("JSON抽出 Demo")
    st.text("文章中から最低月収と最高月収を抽出してJSONにします。月収に関係しない文章ではJSON化しません。")

    json_definition = st.text_area("取り出したいJSON定義", value="""{
        "minimum_monthly_salary": { 
            "type": "number",
            "description": "文章から読み取れる最低月収(単位は円)"
        },
        "maximum_monthly_salary": {
            "type": "number",
            "description": "文章から読み取れる最高月収(単位は円)"
        }
    }""", height=300)
                                
    # jsonとして読み込めるかチェック
    try:
        json_definition = json.loads(json_definition)
    except json.JSONDecodeError:
        st.error("JSON定義が正しくありません")
        return    

    function_name = "extract"
    functions = [
        {
            "name": function_name, 
            "description": "月収や年収に関係のあるテキストから最低月収と最高月収をJSONを抽出する。", 
            "parameters": {
                "type": "object",
                "properties": json_definition,
                "required": list(json_definition.keys()),
            },
        }
    ]

    text = st.text_input("自由記述の文章", value="この求人は月収20万円から50万円です")
    button = st.button("実行")
    if button: 
        result, function_called = function_calling(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            functions=functions,
            function_name=None,
        )
        if function_called: 
            st.json(result)
        else :
            st.info("月収に当たるものが文章から読み取れませんでした")
            st.write(result)

if __name__ == "__main__":
    main()