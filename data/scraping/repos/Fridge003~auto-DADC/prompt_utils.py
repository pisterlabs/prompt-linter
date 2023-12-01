import logging
import os
from typing import Optional, Sequence, Union
from functools import partial
from multiprocessing import Pool
from rouge_score import rouge_scorer
import openai
import pandas as pd
import backoff


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(api_key: str, **kwargs):
    openai.api_key = api_key
    while True:
        try:
            response = openai.ChatCompletion.create(**kwargs)
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
    
    return response['choices'][0]['message']['content']


def generation_prompt(examples, num_genetated_examples, label, prompt_mode="default"):
    """
    :param examples: A list of (premise, hypothesis, label) tuples
    :return: prompt: A string as prompt
    """
    prompt = ""
    id2label = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}

    if prompt_mode == "default":
        num_prompt_examples = len(examples)
        prompt += "In an NLI task, you are given two sentences. The first sentence is called \'Premise\', while" \
                    " the second sentence is called \'Hypothesis\'. The label determines whether “Hypothesis” is " \
                    " true, false, or undetermined under the condition of “premise”. If the answer is true, label should be \'Entailment\';" \
                    "If the answer is false, label should be \'Contradiction\'; If the answer is undetermined, label should be \'Neutral\'."
        
        prompt += f"Now you are going to generate {num_prompt_examples + num_genetated_examples} example of NLI task with {label} as its label." \
                        "Each example should contain three lines, with the first line being a sentence as 'Premise', " \
                        "the second line being a sentence as 'Hypothesis', and the last line being a sentence as 'Label'." 

        for i, example in enumerate(examples):
            prompt += f"{i+1}.\n" \
                    f"Premise:{example['premise']}\n" \
                    f"Hypothesis:{example['hypothesis']}\n" \
                    f"Label:{id2label[example['label']]}\n"
    
    if prompt_mode == "passage":
        prompt += "In NLI task, you are given one passage and a sentence. The passage is called 'Premise', while the sentence is called 'Hypothesis'." \
                  "If 'Premise' clearly supports 'Hypothesis', the label (answer of this task) should be 'Entailment'; " \
                  "If 'Premise' strongly contradicts 'Hypothesis', the label should be 'Contradiction';" \
                  "If 'Premise' can neither support nor contradict 'Hypothesis', or 'Premise' doesn't mention anything about 'Hypothesis', the label should be 'Neutral'.\n"
        picked_example = examples[0]
        passage = picked_example['premise']
        hypothesis = picked_example['hypothesis']
        label = id2label[picked_example['label']]
        prompt += "Here's a passage:\n" + passage + "\n"
        prompt += f"Now you are given the passage above, please generate {num_genetated_examples+1} hypotheses with '{label}' as label and give your explanations.\n"
        prompt += "Here are the requirements: \n" 
        prompt += "1. Both hypothesis and explanation should be 1 to 2 sentences long.\n" 
        prompt += "2. Generate hypothesis at the first line in the format of 'Hypothesis:...'; Generate explanation at the second line in the format of 'Explanation:...'.\n" 
        prompt += "3. If you are going to generate hypothesis with 'Neutral' as label, please don't write any hypothesis that has strong logic relationship with the passage.\n" 
        prompt += "List of hypothesis:\n"
        prompt += f"1. Hypothesis: {hypothesis}\n"

        return prompt

    return prompt


def critique_prompt(example, prompt_mode="default"):

    prompt = ""

    if prompt_mode == "default":
        prompt += "In an NLI task, you are given two sentences. The first sentence is called \'Premise\', while" \
                    " the second sentence is called \'Hypothesis\'. The label determines whether “Hypothesis” is " \
                    " true, false, or undetermined under the condition of “premise”. If the answer is true, label should be \'Entailment\';" \
                    "If the answer is false, label should be \'Contradiction\'; If the answer is undetermined, label should be \'Neutral\'."
        prompt += f"Now you are given an NLI task example, with the \'Premise\' being \'{example['premise']}\', " \
                  f"and the \'Hypothesis\' being \'{example['hypothesis']}\'. Please predict the label."
        prompt += "The predicted label must among one of 'Entailment', 'Contradiction' and 'Neutral'." \
                  "You should predict 'Neutral' when premise doesn't mention anything about 'hypothesis'." \
                  "Give your label at the first line and start another line to explain your answer.\n" \
                  "Label:"
        return prompt
    
    if prompt_mode == "passage":
        prompt += "In NLI task, you are given one passage and a sentence. The passage is called 'Premise', while the sentence is called 'Hypothesis'." \
                  "If 'Premise' clearly supports 'Hypothesis', the label (answer of this task) should be 'Entailment'; " \
                  "If 'Premise' strongly contradicts 'Hypothesis', the label should be 'Contradiction';" \
                  "If 'Premise' can neither support nor contradict 'Hypothesis', or 'Premise' doesn't mention anything about 'Hypothesis', the label should be 'Neutral'.\n"
        prompt += "Here's a passage:\n" + example['premise'] + "\n"
        prompt += f"\nNow you are given the passage as premise above," 
        prompt += f"please predict the label if the hypothesis is '{example['hypothesis']}'." 
        prompt += "The predicted label must among one of 'Entailment', 'Contradiction' and 'Neutral'." \
                  "You should predict 'Neutral' when premise doesn't mention anything about 'hypothesis'." \
                  "Give your label at the first line and start another line to explain your answer.\n" \
                  "Label:"

    return prompt
    

    


def parse_response(response: str, prompt_mode="default") -> Sequence[dict]:
    """
    :param response: a string of response from gpt3/chatgpt
           prompt_mode: method of prompting
    :return: a list of examples int the form of {'premise':.., 'hypothesis':.., 'label':..}
             where label should be 0, 1 or 2
    """

    split_sentences = response.split('\n')
    label2id = {'Entailment': 0, 'Neutral': 1, 'Contradiction': 2}
    collected_examples = []

    if prompt_mode == "default":

        # Assume the response under default mode is in the form of 
        # 1.Premise:...
        # Hypothesis:...
        # Label:...
        # 2. Premise:...
        # Hypothesis:...
        # Label:...
        # ...

        i = 0
        while i < len(split_sentences):

            # Searching for the next example
            if (split_sentences[i].find('Premise') == -1) and \
                (split_sentences[i].find('premise') == -1):
                i += 1
                continue

            if (i + 2 >= len(split_sentences)):
                break

            premise = split_sentences[i][split_sentences[i].find(':')+1:].strip('"')
            hypothesis = split_sentences[i+1][split_sentences[i+1].find(':')+1:].strip('"')
            label = split_sentences[i+2][split_sentences[i+2].find(':')+1:]
            label = label.strip(' .')
            i += 3   
            if label not in label2id.keys():
                continue
            collected_examples.append({"premise": premise, 
                                    "hypothesis": hypothesis,
                                    "label": label2id[label]})
            
    if prompt_mode == "passage":

        # Assume the response is in the form of 
        # 1. Hypothesis:...
        # Explanation:...
        # 2. Hypothesis:...
        # Explanation:...
        # ...

        i = 0
        while i < len(split_sentences):

            # Searching for the next example
            if (split_sentences[i].find('Hypothesis') == -1):
                i += 1
                continue

            if (i + 1 >= len(split_sentences)):
                break
            
            hypothesis = split_sentences[i][split_sentences[i].find(':')+1:]
            i += 1
            collected_examples.append({"premise": None, 
                                    "hypothesis": hypothesis,
                                    "label": None})        
        
    return collected_examples    


def validate_example(example: dict, scorer: rouge_scorer.RougeScorer, all_example_tokens: Sequence, 
                     prompt_args: dict, disagreed_examples: Sequence, num_cpus: int=4, prompt_mode: str="default") -> bool:
    
    id2label = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}

    premise, hypothesis = example["premise"], example["hypothesis"]
    if (len(premise) == 0 or len(hypothesis) == 0):
        return False

    # computing similarity with the pre-tokenzied examples
    if (len(all_example_tokens) > 0):
        similarity_detector = hypothesis if prompt_mode == "passage" else premise + hypothesis
        new_instruction_token = scorer._tokenizer.tokenize(similarity_detector)
        with Pool(num_cpus) as p:
            rouge_scores = p.map(
                partial(rouge_scorer._score_lcs, new_instruction_token),
                all_example_tokens,
            )
        rouge_scores = [score.fmeasure for score in rouge_scores]
        if max(rouge_scores) > 0.7: # There exists some simliar examples
            return False
    
    # Check correctness of example by prompting ChatGPT.
    # If ChatGPT doesn't return the same label as example provides, invalidate this example.
    prompt_for_checking_correctness = critique_prompt(example, prompt_mode)
    prompt_args["temperature"] = 0.2
    prompt_args["messages"] = [{"role":"user", "content": prompt_for_checking_correctness}]
    response = completions_with_backoff(**prompt_args)
    predictied_label = response.split('\n')[0]
    if predictied_label != id2label[example["label"]]:
        example["label"] = f"Generated Label:{id2label[example['label']]}/Label predicted by critic:{predictied_label}"
        disagreed_examples.append(example)
        return False

    return True


# In this function, dataset is stored as a list of dict, 
# where each dict represents one example in the form of {"premise":.., "hypothesis":.., "label":..}.
def load_csv_file_as_list(file_path: str) -> Sequence[dict]:
    list_of_data = []
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        list_of_data += [
            {"premise": df.loc[id, "premise"], 
             "hypothesis": df.loc[id, "hypothesis"], 
             "label": df.loc[id, "label"]}
            for id in range(len(df))
        ]
    return list_of_data



def save_list_as_csv_files(file_path: str, list_of_data: Sequence[dict]):
    df = pd.DataFrame({"premise": [ex["premise"] for ex in list_of_data],
                       "hypothesis": [ex["hypothesis"] for ex in list_of_data],
                       "label": [ex["label"] for ex in list_of_data]})  
                       
    with open(file_path, 'w') as f_out:
        f_out.write(df.to_csv(index=False))
    