import re
import guidance
import pandas
import os
from tqdm import tqdm


def judge(question, answer, predict):
    guidance.llm = guidance.llms.OpenAI(
        "gpt-3.5-turbo",
        api_base="https://api.qaqgpt.com/v1",
        api_key="sk-eKQdkM38VLFqhyz1C2Bf4dB65a9745F09019104515D91422",
    )

    splited_pred = re.split(r"答案[:：]", predict)
    predict = splited_pred[-1].strip()

    # extract all capital letters
    ans_letter = re.sub("[^A-Z]", "", answer)
    ans_letter = set(ans_letter)
    pred_letter = re.sub("[^A-Z]", "", predict)
    pred_letter = set(pred_letter)
    if ans_letter == pred_letter:
        return "YES"
    elif pred_letter.issubset(ans_letter):
        return "PARTIAL"
    else:
        pass  # fall through to the next case with AI

    prompt = guidance(
        """
{{#user~}}
你是TeacherBot，请判断下列学生的回答是否与问题的标准答案相符，即学生给出的答案是否正确。
# 问题：
```
{{question}}
```

# 标准答案：
```
{{answer}}
```

# 学生回答：
```
{{predict}}
```

若学生的回答选择了错误选项，请输入“错误”。
若学生回答没有选择错误选项，但也没有选择所有的正确选项，请输入“不全”。
若学生回答与正确答案完全相符，即选择了所有正确选项，并且未选择任何错误选项，请输入“正确”。
请只回复“错误”、“不全”、“正确”中的一个，不要回复其他内容。
{{~/user}}
                    
{{#assistant~}}
{{gen 'judge' temperature=0 max_tokens=10}}
{{~/assistant}} 
"""
    )

    response = prompt(question=question, answer=answer, predict=predict, caching=False)
    return response["judge"]


if __name__ == "__main__":
    input_path = r"data\PKG_Human_ChatGPT\test.csv"

    file_name_wo_ext = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(
        os.path.dirname(input_path), file_name_wo_ext + "_judged.csv"
    )

    df = pandas.read_csv(input_path)
    if "judge" not in df.columns:
        df["judge"] = None
    for index, row in tqdm(df.iterrows(), desc="Judging", total=len(df)):
        # if not judged
        if pandas.isna(row["judge"]):
            j = judge(row["question"], row["answer"], row["prediction"])
            df.at[index, "judge"] = j
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

            tqdm.write(f"Judged {index}: {j}")
        else:
            tqdm.write(f"Skipped {index}")
