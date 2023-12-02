import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

CHATGPT_API_KEY = ""
CLAUDE_API_KEY = ""
OPEN_AI_MODEL = "gpt-3.5-turbo"
ANTHROPIC_AI_MODEL = "claude-instant-1"
openai.api_key = CHATGPT_API_KEY
anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=CLAUDE_API_KEY,
)


def generate_chatgpt(prompt, open_ai_model=OPEN_AI_MODEL):
    response = openai.ChatCompletion.create(
        model=open_ai_model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    # generated_answer = response['choices'][0]['message']['content']
    messeage = ''
    for chunk in response:
        try:
            generated_answer = chunk['choices'][0]['delta']['content']
            print(generated_answer, end="")
            messeage += generated_answer
        except:
            pass
    return messeage


def generate_claude(prompt, model=ANTHROPIC_AI_MODEL):
    completion = anthropic.completions.create(
        model=model,
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
        stream=True
    )
    message = ''
    for i, chunk in enumerate(completion):
        try:
            if i % 30 == 0:
                print(chunk.completion)
            else:
                print(chunk.completion, end="")
            message += chunk.completion
        except:
            pass
    return message


def generate_flan_t5(prompt):
    ...


if __name__ == '__main__':
    generate_chatgpt('I am your brain. I produce your thoughts. I love you')
