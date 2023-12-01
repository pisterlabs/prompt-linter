import openai


def read_ask(input_string):
    import openai
    # 設定 API 金鑰
    openai.api_key = "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"
    openai.api_type = "open_ai"
    # 定義問題
    question = input_string
    # 設定 GPT-3 模型的引擎 ID
    model_engine = "gpt-3.5-turbo"
    # 呼叫 OpenAI API 並取得答案
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0,
    )
    # 顯示答案
    answer = response.choices[0]['message']['content'].strip()
    return answer


def read_ask_azure(input_string):
    # 設定 API 金鑰
    openai.api_type = "azure"
    openai.api_base = "https://openai-prompt-central-us.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = "2ddbe9da78af446394fad9baced77233"
    # 定義問題
    question = input_string
    answer = ""
    try:
        response = openai.ChatCompletion.create(
            engine="gpt35",
            messages=[{"role": "user", "content": question}],
            temperature=0,
            max_tokens=1200,
            stop=None)
        answer = response.choices[0]['message']['content'].strip()
    except Exception as ex:
        print('Exception:' + str(ex))

    return answer


def read_cvss(input_issue_name, input_type):
    # init
    temp_score = 0
    try:
        print("## " + input_issue_name)
        print("")
        temp_str = f"如果網站上發現\"{input_issue_name}\"最可能會有那些資安問題, 這些資安問題的CVSSv3.1分數數字範圍是多少?"
        print("### 提問: ")
        print("")
        print(temp_str)
        if input_type == 'azure':
            temp_result = read_ask_azure(temp_str)
        else:
            temp_result = read_ask(temp_str)
        print("")
        print("### 回答: ")
        print("")
        print(temp_result)
        print("")
        temp_score_list = []
        for i in range(0, len(temp_result)):
            if temp_result[i] == '.' and temp_result[i + 1] != ' ':
                temp_spilt_score = int(float(temp_result[i - 1] + temp_result[i] + temp_result[i + 1]) * 10)
                # bypass CVSSv3.1
                if temp_spilt_score == 31 and temp_result[i - 2] == 'v':
                    continue
                if temp_spilt_score == 0:
                    if temp_result[i - 1] == '0' and temp_result[i + 1] == '0':
                        temp_spilt_score = 100
                temp_score_list.append(temp_spilt_score)
            if temp_result[i] == '0':
                if temp_result[i - 1] == '1':
                    temp_spilt_score = 100
                    temp_score_list.append(temp_spilt_score)
                elif temp_result[i - 1] != '.':
                    temp_spilt_score = 0
                    temp_score_list.append(temp_spilt_score)

        if len(temp_score_list) == 0:
            temp_score = 0
            print("### 分數:")
            print("")
            print("0(無法評估)")
            print("")
            print("---")
            print("")
        else:
            average = sum(temp_score_list) / len(temp_score_list)
            temp_score = round(average, 1)
            print("### 分數:")
            print("")
            print(str(temp_score) + '(' + str(temp_score_list) + ')')
            print("")
            print("---")
            print("")
    except Exception as ex:
        print('Exception:' + str(ex))
    return temp_score


def read_owasp(input_issue_name):
    # init
    temp_result = ""
    temp_str = f"依序列出2017年 OWASP TOP 10的項目名稱與編號, {input_issue_name}問題屬於哪一種,編號是多少?"
    temp_answer = read_ask(temp_str)
    temp_str = temp_answer[-1]
    if "A1" in temp_str:
        temp_result = "A03"
    elif "A2" in temp_str:
        temp_result = "A07"
    elif "A3" in temp_str:
        temp_result = "A02"
    elif "A4" in temp_str:
        temp_result = "A05"
    elif "A5" in temp_str:
        temp_result = "A01"
    elif "A6" in temp_str:
        temp_result = "A05"
    elif "A7" in temp_str:
        temp_result = "A03"
    elif "A8" in temp_str:
        temp_result = "A08"
    elif "A9" in temp_str:
        temp_result = "A06"
    elif "A10" in temp_str:
        temp_result = "A09"
    return temp_result


if __name__ == '__main__':
    test_issue_name = "任意檔案上傳"
    # print("CVSSv3: " + read_cvss(test_issue_name))
    print("open ai")
    read_cvss(test_issue_name, 'openai api')
    print("azure")
    read_cvss(test_issue_name, 'azure')
    # print("OWASP Top 10(2021): " + read_owasp(test_issue_name))
    # read_owasp(test_issue_name)
    # print(read_ask_azure("說明一下台灣"))
