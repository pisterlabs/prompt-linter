from openai import OpenAI
client = OpenAI(
    # Defaults to os.environ.get("OPENAI_API_KEY")
    # Otherwise use: api_key="Your_API_Key",
)

chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}]
)