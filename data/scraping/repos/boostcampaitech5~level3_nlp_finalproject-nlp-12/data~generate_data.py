import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from tqdm import tqdm


def generate_conversation(
    input_prompt_path: str,
    output_csv_path: str,
    model_name: str = "gpt-3.5-turbo",
    generation_count: int = 1,
    temperature: float = 0.7,
) -> None:
    """OpenAI API를 사용한 멀티-턴 대화 데이터 생성 함수.

    Args:
        input_prompt_path (str): 데이터 생성을 위해 사용할 프롬프트의 경로.
            기본값 "data/input_prompt.txt".
        output_path (str): 생성 데이터를 저장할 경로.
            기본값 "data/output_conversation.csv".
    """
    with open(input_prompt_path, 'r') as f:
        prompt = f.read()
    prompt = prompt + "Provide them in JSON format with the following key: instruction, output."

    for _ in tqdm(range(generation_count)):
        chat_open_ai = ChatOpenAI(model_name=model_name, temperature=temperature)
        response = chat_open_ai.predict(prompt)
        json_data = json.loads(response)

        # Creating pandas DataFrame from json
        df = pd.DataFrame(json_data)

        # Check if file exists to determine whether header is needed
        if os.path.isfile(output_csv_path):
            df.to_csv(output_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_csv_path, mode='w', header=True, index=False)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate data using OpenAI API")
    parser.add_argument("--input_prompt_path", type=str, default="data/input_prompt.txt",
                        help="Path to the prompt template file")
    parser.add_argument("--output_csv_path", type=str, default="data/output_conversation.csv",
                        help="Path to the output file")
    parser.add_argument("--generation_count", type=int, default=1,
                        help="API call count")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                        help="GPT model")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Randomness of conversation data to be generated")
    args = parser.parse_args()

    generate_conversation(
        args.input_prompt_path,
        args.output_csv_path,
        args.model_name,
        args.generation_count,
        args.temperature,
    )


if __name__ == "__main__":
    main()
