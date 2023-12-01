import openai
import argparse
import json
import subprocess
import itertools
import os


prompt1 = """
what are the hyperparameters that can be tuned in the following script? list all the hyperparameters that can potentially be modified along with up to 3 potential values to try for each hyperparameter in the following json format:

{
    "hyperparameter 1": [1, 2, 3],
    "hyperparameter 2": [0.2, 0.5, 0.8]
}

Reply with nothing else except for the json.

Script:
"""

prompt2 = """
Modify the following machine learning script by updating the following hyperparameter. reply with only the updated script and nothing else.
"""

prompt3 = """
You will be given a portion of terminal output from a machine learning script. 
Your task is to output the numerical value of the loss, accuracy or other metric, and to state whether a larger value for this metric is better (True or False).

Example output 1: 
{
    "metric_name": "accuracy",
    "metric_value": 0.386,
    "bigger_is_better": true
}

Example output 2: 
{
    "metric_name": "loss",
    "metric_value": 4.866,
    "bigger_is_better": false
}

Output:
"""


class OAIModel:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.num_retries = 10
        openai.api_key = api_key

    def run(self, prompt: str, output_format: str = 'json'):
        for i in range(self.num_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': 'You are a coding assistant designed to help train machine learning models'},
                        {'role': 'user', 'content': prompt}
                    ])
                response_text = response['choices'][0]['message']['content']
                if output_format == 'json':
                    response_text = response_text.replace(
                        'False', 'false').replace('True', 'true')
                    print(response_text)
                    response_json = json.loads(response_text)

                    return response_json
                else:
                    return response_text
            except Exception as e:
                pass


def argmax(lst, func: max):
    return func(range(len(lst)), key=lambda i: lst[i])


def run_auto_hparam_search(model, file_name):
    with open(file_name, 'r') as file:
        data = file.read()

    hparams = model.run(prompt1 + data, 'json')
    hparams_queue = []
    metrics = []
    hparams_queue = []
    num_retries = 5
    max_runs = 10

    if not os.path.isdir('attempts'):
        os.mkdir('attempts')

    keys, values = zip(*hparams.items())
    for bundle in itertools.product(*values):
        d = dict(zip(keys, bundle))
        hparams_queue.append(dict(zip(keys, bundle)))

    for i, d in enumerate(hparams_queue):

        if i+1 == max_runs:
            break
        _list_str = ''.join([f'_{k}_{v}_' for k, v in d.items()])
        print(f'########## Attempt {i+1}: ' +
              _list_str + ' ##########')
        new_script = model.run(prompt2 + str(d) + '\n\n' + data, 'text')
        new_script_name = 'attempt' + _list_str + '.py'
        for i in range(num_retries):
            try:
                with open(os.path.join('attempts', new_script_name), 'w') as new_file:
                    new_file.write(new_script)
                output = subprocess.check_output(
                    ['python3', os.path.join('attempts', new_script_name)])
                break
            except subprocess.CalledProcessError as e:
                continue
        metrics.append(model.run(prompt3 + str(output)))

    arg_i = argmax([metric['metric_value'] for metric in metrics],
                   max if metrics[0]['bigger_is_better'] else min)

    for metric, d in zip(metrics, hparams_queue):
        print(
            f'for attempt with ' + ''.join([f'{k}: {v} ' for k, v in d.items()]) + f', result: {metric["metric_name"]}: {metric["metric_value"]}')

    print(f'best attempt: {metrics[arg_i]["metric_name"]}: {metrics[arg_i]["metric_value"]} with ' +
          ''.join([f'{k}: {v} ' for k, v in hparams_queue[arg_i].items()]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='AutoHParamSearch',
        description='What the program does',
        epilog='')
    parser.add_argument('--input_file',
                        type=str,
                        help='A script to run the hyperparameter tuning on. Must output some sort of loss or accuracy metric.')
    parser.add_argument('--model_name',
                        type=str,
                        default="gpt-3.5-turbo",
                        help='the chat completions endpoint to use. can be gpt-3.5-turbo, gpt-4, etc.')
    openai_key = os.getenv('OPENAI_KEY')
    if openai_key is None:
        raise ValueError('Please set the OPENAI_KEY environment variable')
    args = parser.parse_args()
    model = OAIModel(args.model_name, openai_key)
    run_auto_hparam_search(model, args.input_file)
