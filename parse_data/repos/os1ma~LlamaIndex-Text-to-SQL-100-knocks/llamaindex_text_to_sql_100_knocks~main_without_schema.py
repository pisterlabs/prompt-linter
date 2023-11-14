from typing import List

import langchain
from extract_100knocks_qa import extract_questions
from langchain.chat_models import ChatOpenAI
from llama_index import LLMPredictor, Prompt
from ruamel.yaml import YAML

verbose = True

# LlamaIndexが内部で使用しているプロンプトから、以下の2行を削除したプロンプト
#
# Only use the tables listed below.
# {schema}
#
# 参考: https://github.com/jerryjliu/llama_index/blob/c60b9420b67cff3b07d483dda96544992d298c6f/gpt_index/prompts/default_prompts.py#L173
template = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed.
Use the following format:
Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Question: {query_str}
SQLQuery: """


class CustomPrompt(Prompt):
    input_variables: List[str] = ["query_str", "dialect"]


def predict(question: str, max_tokens: int):
    # max_tokensを指定しないと、非常に長い時間がかかるため、max_tokensを指定
    llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                     temperature=0, max_tokens=max_tokens)
    predictor = LLMPredictor(llm=llm)
    prompt = CustomPrompt(template)

    response_str, _ = predictor.predict(
        prompt, dialect="postgresql", query_str=question)

    stop_token = "\nSQLResult:"
    stop_token_index = response_str.find(stop_token)

    if stop_token_index == -1:
        error = "stop_token not found"
        qa = {
            'question': question,
            'error': error,
        }
    else:
        answer = response_str[:stop_token_index]
        qa = {
            'question': question,
            'answer': answer,
        }
    return qa


def main():
    if verbose:
        langchain.verbose = True

    questions = extract_questions()

    yaml = YAML()
    yaml.default_style = '|'
    with open('results/result_without_schema.yaml', 'w', encoding='utf-8') as f:

        # text-to-SQLを実行
        for question in questions:
            qa = predict(question, max_tokens=200)

            # エラーになった場合にmax_tokensを追加してリトライ
            if 'error' in qa:
                qa = predict(question, max_tokens=500)

            yaml.dump([qa], f)


if __name__ == "__main__":
    main()
