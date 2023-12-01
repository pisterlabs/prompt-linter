import guidance
import re
import pandas as pd
from tqdm import tqdm


# Assuming the predict function takes a question string as input and returns a prediction
def predict(question):
    # Placeholder implementation; replace with actual prediction logic
    return "Sample Prediction"


def qwen(data_path, output_path="data/zeroshot/qwen.csv"):
    df = pd.read_csv(data_path, encoding="utf-8-sig")

    guidance.llm = guidance.llms.OpenAI(
        "gpt-3.5-turbo",
        api_base="http://a318:8000/v1",
        api_key="",
    )
    prompt = guidance("""
    {{#user~}}
    问题：{{question}}
                       
    请按下列格式回答该问题：
    思考：（对题目的分析和思考过程）
    答案：（题目的答案，只需要给出选项序号即可）
    {{~/user}}
                      
    {{#assistant~}}
    {{gen 'answer' temperature=0 max_tokens=500}}
    {{~/assistant}} 
    """
    )

    def predict(question):
        response = prompt(question=question, caching=False)
        print(response["answer"])
        return response["answer"]

    # Apply the predict function to each question and store the result in the "prediction" column
    tqdm.pandas(desc="Predicting answers")
    df["prediction"] = df["question"].progress_apply(predict)

    # Save the modified dataframe to the specified output_path
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    data_path = r"data\questions\500_embed_sorted.csv"
    qwen(data_path)
