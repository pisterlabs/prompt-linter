import json
import pathlib
import time
import argparse

from base import openai, SLEEP_SECONDS


def read_file_content(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def ask_model_to_generate_code(
        dataset_path,
        code_representation_base_path,
        prompt_path,
        engine,
        temperature,
):
    prompt = read_file_content(prompt_path)

    with open(dataset_path, 'r') as aggregated_boxes_file:
        aggregated_boxes = aggregated_boxes_file.readlines()

    for json_str in aggregated_boxes:
        data = json.loads(json_str)
        sentence = data['sentence']
        sentence_hash = data['sentence_hash']

        code_representation_path = pathlib.Path(f"{code_representation_base_path}/{sentence_hash}.py")

        if not code_representation_path.is_file():
            code_representation_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=[
                        {"role": "user", "content": prompt.format(sentence=sentence)},
                    ],
                    temperature=temperature,
                )
                output = response['choices'][0]['message']['content']

                with open(code_representation_path, 'w') as f:
                    f.write(output)
                print(sentence_hash, "finished")
            except openai.error.OpenAIError as e:
                print(e)
                print("sleeping")
                time.sleep(SLEEP_SECONDS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate code representation using OpenAI.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the input aggregated data.')
    parser.add_argument('--code_representation_base_path', type=str, required=True,
                        help='Base path for code representations.')
    parser.add_argument('--prompt_path', type=str, required=True,
                        help='Path to a prompt.')
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo", help='OpenAI engine to use.')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature setting for OpenAI model.')

    args = parser.parse_args()

    ask_model_to_generate_code(
        args.dataset_path,
        args.code_representation_base_path,
        args.prompt_path,
        args.engine,
        args.temperature,
    )
