import openai

def get_chatgpt_answer(prompt, model_name='gpt-3.5-turbo', role='user'):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{
            'role': role, # user/assistant (mean ChatGPT)/system
            'content': prompt
        }]
    )
    
    return completion['choices'][0]['message']['content']

def apply_prompt(input, prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
        
    return prompt.format(input)