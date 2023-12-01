from airflow.models.variable import Variable
import openai
import pandas as pd
openai.api_key  =  Variable.get("gpt_api_key")
Target_list  =  Variable.get("Target_list")
values = [tuple(item.strip("()").split(",")) for item in Target_list.split("),")]
values = [(x[0].strip(), x[1].strip()) for x in values]
err_report = []
for val in values:

    gpt_ans = []

    temp_df = pd.read_csv(f'/home/jhy/code/TradeTrend/data/{val[0]}/{val[0]}_temp4.csv')
    raw_df = pd.read_csv(f'/home/jhy/code/TradeTrend/data/{val[0]}/{val[0]}_news_raw2.csv')

    ans_list = raw_df.iloc[:, 1]
    while True:
        condition_satisfied = True  # 모든 조건이 만족되었는지 여부를 추적하는 플래그 변수
        for i, ans in enumerate(ans_list):
            try:
                if len(str(ans)) > 4 or (float(ans) > 1 or float(ans) < 0):
                    messages = []
                    a = temp_df.iloc[i, 1]
                    content = f'{a} {val[1]} 관련 뉴스기사 제목인데 {val[1]} 주식에 미칠 긍정도의 평균을 0에서 1사이 소숫점 두자리까지 나타내 float값만'
                    messages.append({"role": "user", "content": content})

                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages
                    )

                    chat_response = completion.choices[0].message.content
                    gpt_ans.append(chat_response)
                    messages.append({"role": "assistant", "content": chat_response})

                    # raw_df에서 해당 값을 새로운 값으로 업데이트합니다.
                    raw_df.iloc[i, 1] = chat_response
                    raw_df.to_csv(f'/home/jhy/code/TradeTrend/data/{val[0]}/{val[0]}_news_raw2.csv', index=False)
                    
                    condition_satisfied = False  # 조건이 하나 이상의 항목에 대해 만족되지 않았음을 표시합니다.
            except: # 에러 발생
                print(i, ans)
                err_report.append(ans)
                condition_satisfied = False
        if condition_satisfied:
            break  # 모든 항목의 조건이 만족되었을 경우 반복문을 종료합니다.
        for err in err_report:
            if err_report.count(err) >=5:
                print("5회 이상 같은 err 발생")
                break

