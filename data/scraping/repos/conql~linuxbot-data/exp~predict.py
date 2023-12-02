import json
import guidance
import re
import pandas as pd
from tqdm import tqdm
import os


def get_prompt():
    with open("data/predict_prompt.txt", "r", encoding="utf-8") as file:
        content = file.read()
        return guidance(content)


def ZERO_SHOT_QWEN():
    llm = guidance.llms.OpenAI(
        "gpt-3.5-turbo",
        api_base="http://a318:8000/v1",
        api_key="",
    )

    prompt = get_prompt()

    def predict(row):
        question = row["question"]
        response = prompt(question=question, attachment="", caching=False, llm=llm)
        return response["answer"]

    return predict


def ZERO_SHOT_CHATGPT():
    llm = guidance.llms.OpenAI(
        "gpt-3.5-turbo",
        api_base="https://api.qaqgpt.com/v1",
        api_key="sk-eKQdkM38VLFqhyz1C2Bf4dB65a9745F09019104515D91422",
    )

    prompt = get_prompt()

    def predict(row):
        question = row["question"]
        response = prompt(question=question, attachment="", caching=False, llm=llm)
        return response["answer"]

    return predict


def PKG_GPT4():
    llm = guidance.llms.OpenAI(
        "gpt-3.5-turbo",
        api_base="http://a318:8000/v1",
        api_key="",
    )

    prompt = get_prompt()

    def predict(row):
        knowledge = json.loads(row["knowledge"])
        question = row["question"]
        attachment = (
            "已知：\n"
            + "\n".join([f'【{kp["title"]}】\n{kp["content"]}' for kp in knowledge])
            + "\n"
        )

        response = prompt(
            question=question, attachment=attachment, caching=False, llm=llm
        )
        return response["answer"]

    return predict


def PKG_Human():
    llm = guidance.llms.OpenAI(
        "gpt-3.5-turbo",
        api_base="http://a318:8000/v1",
        api_key="",
    )

    prompt = get_prompt()

    def predict(row):
        if not pd.isna(row["perfect_knowledge"]):
            knowledge = json.loads(row["perfect_knowledge"])
            tqdm.write("Using perfect knowledge")
        else:
            knowledge = json.loads(row["knowledge"])

        question = row["question"]
        attachment = (
            "已知：\n"
            + "\n".join([f'【{kp["title"]}】\n{kp["content"]}' for kp in knowledge])
            + "\n"
        )

        response = prompt(
            question=question, attachment=attachment, caching=False, llm=llm
        )
        return response["answer"]

    return predict


def PKG_Human_ChatGPT():
    llm = guidance.llms.OpenAI(
        "gpt-3.5-turbo",
        api_base="https://api.qaqgpt.com/v1",
        api_key="sk-eKQdkM38VLFqhyz1C2Bf4dB65a9745F09019104515D91422",
    )

    prompt = get_prompt()

    def predict(row):
        if not pd.isna(row["perfect_knowledge"]):
            knowledge = json.loads(row["perfect_knowledge"])
            tqdm.write("Using perfect knowledge")
        else:
            knowledge = json.loads(row["knowledge"])

        question = row["question"]
        attachment = (
            "已知：\n"
            + "\n".join([f'【{kp["title"]}】\n{kp["content"]}' for kp in knowledge])
            + "\n"
        )

        response = prompt(
            question=question, attachment=attachment, caching=False, llm=llm
        )
        return response["answer"]

    return predict


if __name__ == "__main__":
    data_path = r"data\questions\test.csv"
    method = "PKG_Human_ChatGPT"

    output_dir = os.path.join("data", method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name_wo_ext = os.path.splitext(os.path.basename(data_path))[0]
    output_path = os.path.join(output_dir, f"{file_name_wo_ext}.csv")

    predict = globals()[method]()

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    if "prediction" not in df.columns:
        df["prediction"] = None

    for index, row in tqdm(df.iterrows(), desc="Predicting answers", total=len(df)):
        if not pd.isna(row["prediction"]):
            continue
        else:
            prediction = predict(row)
            df.at[index, "prediction"] = prediction
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            tqdm.write(f"Predicted {index}: \n{prediction}")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
