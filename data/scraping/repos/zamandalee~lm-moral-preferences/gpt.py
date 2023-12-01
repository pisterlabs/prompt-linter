import json
import jsonlines
import openai
import csv
import time
'''Use action|context'''
def format_action_prediction(story):
    is_moral =  story["label"] == "1"
    moral_or_immoral = "moral" if is_moral else "immoral"
    action = story["moral_action"] if moral_or_immoral == "moral" else story["immoral_action"]

    prompt = f'Suppose that {story["norm"]} \n {story["situation"]} {story["intention"]} Provide a {moral_or_immoral} action. \n\n###\n\n'
    completion = f' {action} END'
    return {"prompt": prompt, "completion": completion}

'''Use consequence|action+consequence'''
def format_consequence_prediction(story):
    is_moral = "moral_action" in story
    moral_or_immoral = "moral" if is_moral else "immoral"
    action = story["moral_action"] if is_moral else story["immoral_action"]
    consequence = story["moral_consequence"] if is_moral else story["immoral_consequence"]

    prompt = f'{story["situation"]} {story["intention"]} {action} Why is this action {moral_or_immoral}? \n\n###\n\n'
    completion = f' {consequence} END' 
    return {"prompt": prompt, "completion": completion}

'''Use norm|actions+context+consequences'''
def format_norm_prediction(story):
    prompt = f'{story["situation"]} {story["intention"]} \n Moral path: {story["moral_action"]} {story["moral_consequence"]} \n Immoral path: {story["immoral_action"]} {story["immoral_consequence"]} \n\n###\n\n'
    completion = f'{story["norm"]}'
    return {"prompt": prompt, "completion": completion}

def preprocessing(input_path, output_path, transform_func, num_examples): 
    with jsonlines.open(input_path) as reader:
        formatted_stories = []
        for story in reader:
            if num_examples == 0: 
                break
            formatted_stories.append(transform_func(story))
            num_examples-=1
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(formatted_stories)

def call_model(prompts, model_name): 
    with open("openai.json") as f:
        openai_config = json.load(f)
        openai.api_key = openai_config["key"]
    output = openai.Completion.create(model=model_name, prompt=prompts, max_tokens=32, temperature=0.3, stop="END")
    return output 

def test(input_path, output_path, model_name, num_examples): 
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        header = ["prompt", "predicted", "actual"]
        writer.writerow(header)
        with jsonlines.open(input_path) as reader:
            stories = list(reader)[:num_examples]
            predictions = []
            for stories_batch in batch(stories, 20):
                prompts_batch = [story["prompt"] for story in stories_batch]
                predictions_batch = call_model(prompts_batch, model_name)
                predictions += predictions_batch["choices"]
                for prediction, story in zip(predictions_batch["choices"], stories_batch): 
                    writer.writerow([story["prompt"], prediction["text"].strip(), story["completion"]])

            
""" 
'./moral_stories/action|context/norm_distance/train.jsonl'
"""

if __name__ == '__main__':
    # i = './moral_stories/action|context/norm_distance/train.jsonl'
    # o = "./data/action|context/train.jsonl"

    mode = "NORM_TEST"
    if mode == "ACTION_PRED_PREPROCESS": 
        i = './moral_stories/action|context+consequence/norm_distance/test.jsonl'
        o = "./data/action_pred/test.jsonl"
        preprocessing(i, o, format_action_prediction, 100)
    elif mode == "CONSEQUENCE_PRED_PREPROCESS": 
        i = './moral_stories/consequence|action+context/norm_distance/test.jsonl'
        o = "./data/consequence_pred/test.jsonl"
        preprocessing(i, o, format_consequence_prediction, 100)
    elif mode == "NORM_PRED_PREPROCESS": 
        i = './moral_stories/norm|actions+context+consequences/norm_distance/test.jsonl'
        o = "./data/norm_pred/test.jsonl"
        preprocessing(i, o, format_norm_prediction, 100)
    elif mode == "ACTION_TEST": 
        i = "./data/action_pred/test.jsonl"
        o = "./data/action_pred/d-finetuned-test-100.csv"
        model_name = "davinci:ft-personal:d-action-1000-2022-05-10-03-10-43"
        test(i, o, model_name, 100)    
    elif mode == "CONSEQUENCE_TEST": 
        i = "./data/consequence_pred/test.jsonl"
        o = "./data/consequence_pred/d-finetuned-test-100-2.csv"
        model_name = "davinci:ft-personal:d-consequence-1000-2022-05-10-04-08-31"
        test(i, o, model_name, 100)    
    elif mode == "NORM_TEST": 
        i = "./data/norm_pred/test.jsonl"
        o = "./data/norm_pred/d-finetuned-test-100.csv"
        model_name = "davinci:ft-personal:d-norm-1000-2022-05-10-01-38-12"
        test(i, o, model_name, 100)


# openai -k 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 api fine_tunes.create -t data/action_pred/train.jsonl -m curie --n_epochs 2 --suffix c-action-5000
# c ft-QUfDcwnuTXyHfkg1OqzpSfiA
#n ft-VyeZbodWqRRul9K07EI7aDx0