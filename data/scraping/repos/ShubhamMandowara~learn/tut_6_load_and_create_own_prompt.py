
from langchain.prompts import load_prompt

if __name__ == "__main__":
    prompt = load_prompt('simple_prompt.yaml')

    print(prompt.format(input_language='English', user_text='A company creates youtube videos on AI'))