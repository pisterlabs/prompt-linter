import anthropic_api
import openai_api
import yaml

with open('config.yml') as f:
    config = yaml.safe_load(f)

def api_call(reviewer_name, content, reviewer_no):
    if 'gpt' == reviewer_name[:3]:
        return openai_api.call(content, reviewer_no)
    elif 'claude' == reviewer_name[:6]:
        return anthropic_api.call(content, reviewer_no)

def result_extraction(text):
    try:
        if config['add_prompt_each_turn']:
            text = text.split('\n\n[System]')[0]
        preference = int(text.strip().split('\n')[-1].strip())
        return preference
    except Exception as e:
        print(e)
        return -1

def history_formatting(history, reviewers):
    prompt = [{'system': history[0]},
              {'background': history[1]}]
    history = history[2:]

    for idx, content in enumerate(history):
        if 0 == idx % 2:
            prompt.append({f'{reviewers[0]}': {
                                'content': content,
                                'preference': result_extraction(content)
                           }})
        else:
            prompt.append({f'{reviewers[1]}': {
                                'content': content,
                                'preference': result_extraction(content)
                           }})
    return prompt

if __name__ == '__main__':
    pass

