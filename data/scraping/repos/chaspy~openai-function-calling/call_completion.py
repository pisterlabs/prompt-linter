from openai import OpenAI

client = OpenAI()  # load OPENAI_API_KEY from environment variable

chat_completions = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "あなたはプロのスタジオミュージシャンです。"},
        {
            "role": "user",
            "content": "初心者がドラムを練習する方法を教えてください"},
    ],
    model="gpt-3.5-turbo",
)
print(chat_completions)
