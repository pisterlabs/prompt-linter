import openai

def generate_text(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        max_tokens=1024,
        n=1,
        stop=None,
    )
    message = completions.choices[0].text
    # 最初の改行は削除
    message = message.replace("\n", "", 2)
    return message

if __name__ == '__main__':
    # secrets/openai_API_KEY.txtにAPIキーを保存して、それを読み込む
    openai.api_key = open("secrets/openai_API_KEY.txt").read().strip()
    print(generate_text("今日はいい天気ですね。"))