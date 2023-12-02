from openai import OpenAI


with open('/home/wsutt/.openai-key.txt', 'r') as f:
    env_key = f.read().strip()
    # TODO - add .env file to store API key
    # defaults to os.environ.get("OPENAI_API_KEY")


def submit_prompt(
    prompt: str,
    model_name: str,
    max_tokens: int = 60,
    temperature: float = 0.7,
) -> dict:

    client = OpenAI(
        api_key=env_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],    
        temperature=temperature,
        max_tokens=max_tokens,
        model=model_name,
    )

    return chat_completion


def get_completion(
    chat_completion: dict,
    role: str = "ai",
) -> str:

    return chat_completion.choices[0].message.content

