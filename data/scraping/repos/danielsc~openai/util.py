import evaluate, yaml
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from tqdm import tqdm
import pandas as pd

def find_first_number(text):
    number = next((c for i, c in enumerate(text) if c.isdigit()), "4")
    parsed_number = int(number)

    if parsed_number < 1:
      number = "1"
    elif parsed_number > 5:
      number = "5"
    return number

def calculate_metrics(references, predictions):
    f1 = evaluate.load("f1").compute(references=references, predictions=predictions, average='weighted')
    accuracy = evaluate.load("accuracy").compute(references=references, predictions=predictions)
    precision = evaluate.load("precision").compute(references=references, predictions=predictions, average='weighted')
    recall = evaluate.load("recall").compute(references=references, predictions=predictions, average='weighted')
    # merge the dicts
    metrics = {**f1, **accuracy, **precision, **recall}
    return metrics

def save_yaml(content, filename):
   with open(filename, encoding='utf-8', mode='w') as fh:
      yaml.dump(content, fh)


def classify_yelp_fsp_batch(llm, 
                            texts, 
                            examples,
                            prefix,
                            template,
                            suffix,
                            batch_size=20):
    results = []
    example_prompt = PromptTemplate(input_variables=["n", "review", "stars"], template=template)

    few_shot_prompt = FewShotPromptTemplate(
        prefix=prefix,
        examples=examples, 
        example_prompt=example_prompt, 
        suffix=suffix, 
        input_variables=["text"]
    )

    for i in tqdm(range(0, len(texts), batch_size)):
        prompts = [ few_shot_prompt.format(text=text) for text in texts[i:i+batch_size]]
        batch_results = llm(prompts)
        batch_results = [find_first_number(item) for item in batch_results]
        results.extend(batch_results)
    return results

def get_examples(examples_file, examples_number, seed, prompt_column, completion_column):
    """
    Selects a number of examples from a file
    within each set of 5 examples, don't have two with the same stars rating so that we have some variety in the prompt
    """
    examples_df = pd.read_csv(examples_file)
    
    examples = []
    bucket = set()
    for i in range(examples_number):
        while True:
            example = examples_df.sample(n=examples_number, ignore_index=True, random_state=seed).iloc[0]
            seed = seed+1
            if not example[completion_column] in bucket:
                bucket.add(example[completion_column])
                break
        examples.append({"n": i+1, "review": example[prompt_column], "stars": example[completion_column]})

        if len(bucket) == 5:
            bucket = set()
    return examples