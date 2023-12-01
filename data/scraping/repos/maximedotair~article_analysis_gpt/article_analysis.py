import openai
from newspaper import Article

def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    title = article.title
    text = article.text

    tokens = text.split()
    max_tokens_per_part = 1000
    text_parts = []
    current_part = []

    for token in tokens:
        current_part.append(token)
        if len(current_part) >= max_tokens_per_part:
            text_parts.append(" ".join(current_part))
            current_part = []

    if current_part:
        text_parts.append(" ".join(current_part))

    return title, text_parts

def chat(api_key, text_messages, user_message):
    openai.api_key = api_key

    messages = text_messages.copy()
    messages.append({"role": "user", "content": user_message})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=2048,
        n=1,
        temperature=0.2,
        stop=None
    )

    answer = response.choices[0].message['content'].strip()
    return answer

def main():
    api_key = "your_openai_api_key"
    url = input("Enter the URL of the article to summarize: ")

    title, text_parts = get_article(url)

    text_messages = [
        {"role": "assistant", "content": "Hello, I am the virtual assistant. How may I help you today?"},
    ]

    for i, part in enumerate(text_parts):
        text_messages.append({"role": "user", "content": f"I will provide you with a text in different parts,once you have read and taken into account say just 'ok' and I will add the following parts, when I talk about the text or the article it is the whole text broken down that you will have to take into account, here is the {i + 1}st: \"{part}\""})

    text_messages.append({"role": "user", "content": f"Now that you have all the parts of the text, I will ask you some questions, I will always refer to the text. Now Summarize this article in a maximum of 2 paragraphs, in English."})

    summary = chat(api_key, text_messages, "Summary:")
    print(f"\nSummary: {summary}")

    while True:
        user_question = input("\nAsk a question (previous questions and answers are not saved) or type 'exit' to quit: ")
        if user_question.lower() == 'exit':
            break
        else:
            answer = chat(api_key, text_messages, user_question)
            print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
