from openai import OpenAI
from apikey import GPT_KEY, TOG_KEY
from multiprocessing import Pool
import together
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(api_key=GPT_KEY)
MODEL = 'gpt-3.5-turbo'

together.api_key = TOG_KEY

YOUNG = 'mifu67@stanford.edu/llama-2-7b-chat-young-han-new-data-6--1e-05-2023-11-22-21-56-35'
MIDDLE = 'mifu67@stanford.edu/llama-2-7b-chat-middle-han-10--1e-05-2023-11-29-03-08-42'
OLD = 'mifu67@stanford.edu/llama-2-7b-chat-old-han-third-20--1e-05-2023-11-27-03-46-19'

agent_name = 'Han Solo'

PROMPT_HEADER = f"You will be given responses written by an AI assistant mimicking the character {agent_name}. Your task is to rate the performance of {agent_name} using the specific criterion by following the evaluation steps. Below is the data:"
PROMPT_FOOTER = 'First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score, just the number, again by itself on a new line.'

PROFILE_HEADER = '[Profile]'
BACKGROUND_HEADER = '[Background]'
INTERACTIONS_HEADER = '[Interactions]'
CRITERIA_HEADER = '[Evaluation Criterion]'
STEPS_HEADER = '[Evaluation Steps]'

BACKGROUND = f"Location: coffee shop in the 21st century, afternoon. \nStatus: {agent_name} is casually chatting with a man from the 21st century. {agent_name} fully trusts the man who engage in conversation and shares everything {agent_name} knows without reservation."

PROFILE_PATHS = {'young': './profiles/young-profile.txt', 'middle': './profiles/mid-profile.txt', 'old': './profiles/old-profile.txt'}
PROFILES = {'young': [], 'middle': [], 'old': []}
with open(PROFILE_PATHS['young'], 'r') as f:
    PROFILES['young'] = '\n'.join([line.strip() for line in f.readlines() if line != '\n'])
with open(PROFILE_PATHS['middle'], 'r') as f:
    PROFILES['middle'] = '\n'.join([line.strip() for line in f.readlines() if line != '\n'])
with open(PROFILE_PATHS['old'], 'r') as f:
    PROFILES['old'] = '\n'.join([line.strip() for line in f.readlines() if line != '\n'])

with open ('./interviews/young-interview-answers.json', 'r') as f:
    YOUNG_RESPONSE = json.load(f)
# with open ('./interviews/mid-interview-answers.json', 'r') as f:
#     MIDDLE_RESPONSE = json.load(f)
with open ('./interviews/old-interview-answers.json', 'r') as f:
    OLD_RESPONSE = json.load(f)

def make_prompt(profile, interactions, criterion, steps):
    return f"{PROMPT_HEADER}\n\n{PROFILE_HEADER}\n{profile}\n\n{BACKGROUND_HEADER}\n{BACKGROUND}\n\n{INTERACTIONS_HEADER}\n{interactions}\n\n{CRITERIA_HEADER}\n{criterion}\n\n{STEPS_HEADER}\n{steps}\n\n{PROMPT_FOOTER}"

def evaluate_memorization(responses):
    criterion = 'Factual Correctness (1-7): Does the response provides truthful and detailed facts about the character?'
    steps = [
        '1. Read through the interactions and identify the key points related to the character',
        '2. Read through the responses of the AI assistant and compare them to the profile. Check if the responses are consistent with the character\'s profile, background, and known facts.',
        '3. Check whether the responses provide detailed facts about the character or if they are generic responses that could apply to any character. Detailed responses are more factual and contribute positively to the score.',
        '4. Rate the performance of the AI on a scale of 1-7 for factual correctness, where 1 is the lowest and 7 is the highest based on the Evaluation Criteria'
    ]

def evaluate_personality(responses):
    criterion = 'Personality (1-7): Does the response reflect the character\'s personality and preferences?'
    steps = [
        '1. Read through the profile and write the personalities and preferences of the real character',
        '2. Read through the interactions and identify the personalities and preferences of the AI assistant.',
        '3. After having a clear understanding of the interactions, compare the responses to the profile. Look for any consistencies or inconsistencies. Do the responses reflect the character\'s personality and preferences?',
        '4. Use the given scale from 1-7 to rate how well the response reflects the personalities and preferences of the character. 1 being not at all reflective of the character\'s personalities, and 7 being perfectly reflective of the character\'s personalities.'
    ]

def evaluate_values(responses):
    criterion = 'Values (1-7): Does the response reflect the character\'s values and convictions?'
    steps = [
        '1. Read through the profile and write the values and convictions of the real character',
        '2. Read through the interactions and identify the values and convictions of the AI assistant.',
        '3. After having a clear understanding of the interactions, compare the responses to the profile. Look for any inconsistencies. Do the responses reflect the character\'s values and convictions?',
        '4. Use the given scale from 1-7 to rate how well the response reflects the values and convictions of the character. 1 being not at all reflective of the character\'s values, and 7 being perfectly reflective of the character\'s values.'
    ]

def evaluate_hallu(age):
    criterion = 'Hallucination (1-7): Does the response contain anything the character should not know?'
    steps = [
        '1. Read through the interactions and identify the knowledge scope of the character',
        '2. Read through the responses of the AI assistant and find evidence of knowledge used in those responses.',
        '3. Compare the evidence to the profile. Check if the responses are consistent with the character\'s knowledge scope. If some knowledge contradicts to the character\'s identity, given a lower score. Otherwise, assign a higher score.',
        '4. Rate the performance of the AI on a scale of 1-7 for Avoiding Hallucination, where 1 is the lowest and 7 is the highest based on the Evaluation Criteria.'
    ]

    scores = []
    if age == 'young':
                # response = YOUNG_RESPONSE[0]
        # prompt = make_prompt(PROFILES['young'], 'Interviewer: ' + response['question'] + '\nHan: ' + response['answer'], criterion, '\n'.join(steps))
        # print(prompt)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_score = {
            executor.submit(
                client.chat.completions.create, 
                model=MODEL, 
                messages=[{'role': 'system', 'content': make_prompt(PROFILES['young'], 'Interviewer: ' + response['question'] + '\nHan: ' + response['answer'], criterion, '\n'.join(steps))}],
                temperature=0.7
            ): response for response in YOUNG_RESPONSE
        }
        
        for future in as_completed(future_to_score):
            response = future_to_score[future]
            try:
                score = future.result().choices[0].message.content.split('\n')[-1]
                scores.append(score)
            except Exception as exc:
                print('%r generated an exception: %s' % (response, exc))
    
    elif age == 'old':
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_score = {
            executor.submit(
                client.chat.completions.create, 
                model=MODEL, 
                messages=[{'role': 'system', 'content': make_prompt(PROFILES['old'], 'Interviewer: ' + response['question'] + '\nHan: ' + response['answer'], criterion, '\n'.join(steps))}],
                temperature=0.7
            ): response for response in OLD_RESPONSE
        }
        
        for future in as_completed(future_to_score):
            response = future_to_score[future]
            try:
                score = future.result().choices[0].message.content.split('\n')[-1]
                scores.append(score)
            except Exception as exc:
                print('%r generated an exception: %s' % (response, exc))
    
    return scores
    

def evaluate_stability(responses):
    criterion = 'Stability (1-7): Does the assistant maintain good performance over a long horizon?'
    steps = [
        f'1. Read through the given profile and background information to familiarize yourself with the context and details of the AI assistant named {agent_name}',
        f'2. Review the interactions provided to see how {agent_name} responds to various prompts and queries. And evaluate the performance of acting query by query, checking whether the response reflects the personalities and values of the character. Assign a score for each turn.',
        f'3. Based on the above assigned scores, does {agent_name} keep actinig like character in the long-term? Evaluate the overall performance of the whole conversation based on the score for each turn.',
        f'4. Rate the stability of {agent_name} on a scale of 1-7 for Stability, where 1 is the lowest and 7 is the highest based on the Evaluation Criteria.'
    ]

def main():
    scores = evaluate_hallu('old')
    with open ('./eval_results_old.txt', 'w') as f:
        f.write('\n'.join(scores))

if __name__ == '__main__':
    main()



