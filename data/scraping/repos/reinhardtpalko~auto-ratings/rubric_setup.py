import argparse
import json
import sys

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.chains import LLMChain

from src.prompt_templates import question2rubric_template
from src.json_validator import validate_questions

load_dotenv()


def main(args):
    file = args.file
    new_file = args.new_file

    with open(file) as f:
        file = json.load(f)

    result = exec(file)

    with open(new_file, "w") as f:
        json.dump(result, f)


def exec(json_obj):
    test = validate_questions(json_obj)
    if not test:
        sys.exit(1)
    else:
        questions = json_obj["questions"]
        scenario = json_obj["scenario"]

        prompt_template = question2rubric_template()

        llm = ChatOpenAI(model="gpt-4-0613", temperature=0)

        chain = LLMChain(llm=llm, prompt=prompt_template)

        new_dict = {"questions": []}

        for q in questions:
            question = q["question"]
            item = {
                "question": question,
                "question_id": q["question_id"],
                "dimension": q["dimension"],
            }
            result = chain.run(scenario=scenario, question=question)
            result = json.loads(result)
            item["rubric"] = result
            new_dict["questions"].append(item)

        return new_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texture Editing")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Directory of the json file containing the questions.",
    )
    parser.add_argument(
        "-n", "--new_file", type=str, help="Directory of the new json file."
    )

    args = parser.parse_args()
    main(args)
