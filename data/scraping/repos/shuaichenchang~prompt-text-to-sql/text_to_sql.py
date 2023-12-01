import openai
import os
import random
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from database_prompt_construction import generate_db_prompt, prompt_length_by_db, OOD_SCHEMA_MAXLEN
from sql_generation import text_to_sql_direct, text_to_sql_few_shot_singledomain, text_to_sql_few_shot_crossdomain
from utils import find_random_examples, db_ids_dataset

DATA_PATH = "data_processed"


def zeroshot(dataset, openai_model="codex", prompt_db="CreateTableSelectCol", limit=0, normalization=True):
    output_path = f"outputs/{openai_model}/{dataset}/zeroshot/{prompt_db}"
    if normalization:
        sql_format = "normalized"
    else:
        sql_format = "unnormalized"
    output_path += f"_{sql_format}"
    if limit > 0:
        output_path += f"_limit_{limit}"
    db_ids = db_ids_dataset[dataset]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    predictions_total = []
    prompt_total = []
    for db_id in db_ids:
        print("=" * 10 + db_id + "=" * 10)
        prompt_template = generate_db_prompt(dataset, db_id, prompt_db=prompt_db, limit_value=limit, normalization=normalization)

        with open(f"{DATA_PATH}/{dataset}/questions/{db_id}.json", "r") as f:
            questions = json.load(f)

        predictions, prompts = text_to_sql_direct(openai_model, questions, prompt_template, sql_format)

        predictions_total.extend(predictions)
        prompt_total.extend(prompts)
        with open(f"{output_path}/{db_id}.json", "w") as f:
            json.dump(predictions, f, indent=4)
        with open(f"{output_path}/prompt_{db_id}.json", "w") as f:
            json.dump(prompts, f, indent=4)
    predictions_total = []

    for db_id in db_ids:
        with open(f"{output_path}/{db_id}.json", "r") as f:
            predictions = json.load(f)
        predictions_total.extend(predictions)
    with open(f"{output_path}/pred.json", "w") as f:
        json.dump(predictions_total, f, indent=4)
    with open(f"{output_path}/prompts.json", "w") as f:
        json.dump(prompt_total, f, indent=4)

    with open(f"{output_path}/pred.sql", "w") as f:
        for d in predictions_total:
            f.write(d["predicted_sql"].replace('\n', ' ') + '\t' + d["db_id"] + '\n')
    with open(f"{output_path}/gold.sql", "w") as f:
        for d in predictions_total:
            f.write(d["gold_sql"].replace('\n', ' ') + '\t' + d["db_id"] + '\n')


def few_shot(dataset, openai_model, setting, prompt_db, limit, num_table=3, num_shot_per_table=5, num_shot=1, seed=12345, normalization=True):
    if normalization:
        demo_sql_format = "normalized"
    else:
        demo_sql_format = "unnormalized"
    dataset_for_input = dataset

    db_ids = db_ids_dataset[dataset_for_input]
    output_path = f"outputs/{openai_model}/{dataset}/{setting}/{prompt_db}_{demo_sql_format}"
    if limit > 0:
        output_path += f"_limit_{limit}"
    output_path += f"_{seed}"
    if setting == "singledomain":
        shot_name = f"shot_{num_shot}"
    elif setting == "crossdomain":
        shot_name = f"table_{num_table}_shot_{num_shot_per_table}"
    else:
        raise NotImplementedError

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if setting == "crossdomain":
        with open(f"{DATA_PATH}/spider-train/questions/questions.json", "r") as f:
            outdomain_questions = json.load(f)
        outdomain_questions = [q for q in outdomain_questions if prompt_length_by_db[q["db_id"]] < OOD_SCHEMA_MAXLEN]

    few_shot_in_prompts = {}
    predictions_total = []
    random.seed(seed)

    for db_id in db_ids:
        with open(f"{DATA_PATH}/{dataset_for_input}/questions/{db_id}.json", "r") as f:
            questions = json.load(f)
        if setting == "singledomain":
            indomain_questions_for_retrieval = questions[:]

        outdomain_createtable_schemas_per_question = []
        outdomain_demo_examples_per_question = []
        indomain_demo_examples_per_question = []

        for i in tqdm(range(len(questions))):
            q = questions[i]
            if setting == "crossdomain":
                outdomain_questions_for_retrieval = find_random_examples(q, outdomain_questions, split=None, deduplicate_demo="nlq")
                examples_per_db = defaultdict(list)
                outdomain_createtable_schemas = []
                outdomain_demo_examples = []
                for retrieval_q in outdomain_questions_for_retrieval:
                    if len(examples_per_db[retrieval_q["db_id"]]) >= num_shot_per_table:
                        continue
                    examples_per_db[retrieval_q["db_id"]].append(retrieval_q)
                    if len(examples_per_db[retrieval_q["db_id"]]) == num_shot_per_table:
                        outdomain_createtable_schemas.append(
                            generate_db_prompt("spider-train", retrieval_q["db_id"], prompt_db, limit_value=limit))
                        outdomain_demo_examples.append(examples_per_db[retrieval_q["db_id"]][::-1])
                        if len(outdomain_createtable_schemas) == num_table:
                            outdomain_createtable_schemas = outdomain_createtable_schemas[::-1]
                            outdomain_demo_examples = outdomain_demo_examples[::-1]
                            break
                outdomain_createtable_schemas_per_question.append(outdomain_createtable_schemas)
                outdomain_demo_examples_per_question.append(outdomain_demo_examples)

            if setting == "singledomain":
                indomain_demo_examples = find_random_examples(q, indomain_questions_for_retrieval, split="template", deduplicate_demo="nlq")
                indomain_demo_examples = indomain_demo_examples[:num_shot]
                indomain_demo_examples = indomain_demo_examples[::-1]
                indomain_demo_examples_per_question.append(indomain_demo_examples)

        indomain_createtable_schema = generate_db_prompt(dataset_for_input, db_id, prompt_db, limit_value=limit, normalization=normalization)

        if setting == "singledomain":
            few_shot_in_prompt, predictions = text_to_sql_few_shot_singledomain(openai_model, questions, indomain_createtable_schema,
                                                                                indomain_demo_examples_per_question, demo_sql_format=demo_sql_format)
        elif setting == "crossdomain":
            few_shot_in_prompt, predictions = text_to_sql_few_shot_crossdomain(openai_model, questions, outdomain_createtable_schemas_per_question,
                                                                               indomain_createtable_schema,
                                                                               outdomain_demo_examples_per_question, demo_sql_format=demo_sql_format)
        else:
            raise "unknown setting"
        predictions_total.extend(predictions)
        few_shot_in_prompts[db_id] = few_shot_in_prompt

        with open(os.path.join(output_path, f"{db_id}_{shot_name}.json"), "w") as f:
            json.dump(predictions, f, indent=4)

    with open(os.path.join(output_path, f"pred_{shot_name}.json"), "w") as f:
        json.dump(predictions_total, f, indent=4)
    with open(os.path.join(output_path, f"pred_{shot_name}.sql"), "w") as f:
        for d in predictions_total:
            f.write(d["predicted_sql"] + '\t' + d["db_id"] + '\n')
    with open(os.path.join(output_path, f"few_{shot_name}.json"), "w") as f:
        json.dump(few_shot_in_prompts, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="spider-dev", type=str, help='dataset', choices=["spider-dev"])
    parser.add_argument('--setting', type=str, help='setting', choices=["zeroshot", "singledomain", "crossdomain"], default="zeroshot")
    parser.add_argument('--model', default="codex", type=str, help='open ai model', choices=["codex", "chatgpt"])
    parser.add_argument('--prompt_db', type=str, help='prompt construction for database', default="CreateTableSelectCol",
                        choices=["Table(Columns)", "Columns=[]", "Columns=[]+FK", "CreateTable", "CreateTableInsertRow", "CreateTableSelectRow",
                                 "CreateTableSelectCol"])

    args = parser.parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    dataset = args.dataset
    setting = args.setting
    openai_model = args.model
    prompt_db = args.prompt_db
    if prompt_db in ["Table(Columns)", "Columns=[]", "Columns=[]+FK", "CreateTable"]:
        database_content_row = 0
    elif prompt_db in ["CreateTableInsertRow", "CreateTableSelectRow", "CreateTableSelectCol"]:
        database_content_row = 3
    else:
        raise "unknown prompt_db"

    if setting == "zeroshot":
        zeroshot(dataset, openai_model=openai_model, prompt_db=prompt_db, limit=database_content_row, normalization=True)

    elif setting == "singledomain":
        for seed in [12345, 12346, 12347]:
            for num_shot in [1, 4, 8, 16]:
                few_shot(dataset, openai_model=openai_model, setting=setting, prompt_db=prompt_db, limit=database_content_row, num_shot=num_shot, seed=seed,
                         normalization=True)

    elif setting == "crossdomain":
        for seed in [12345, 12346, 12347]:
            for (num_table, num_shot_per_table) in [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16)]:
                few_shot(dataset, openai_model=openai_model, setting=setting, prompt_db=prompt_db, limit=3, num_table=num_table,
                         num_shot_per_table=num_shot_per_table, seed=seed, normalization=True)
