import csv
import random
import os
from openai import OpenAI

# 从环境变量获取 API 密钥
api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

def generate_sentence(word1, definition1, word2, definition2, word3, definition3):
    try:
        prompt = f"我正在通过句子学习英文单词，请你给出在一个句子中包含 '{word1}' 和 '{word2}' 和 '{word3}' 的英文，尽量口语化, 然后给出中文翻译，\
    	并为句子中高于初中水平的所有单词提供国际音标、英文解释和中文解释"
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a english teacher."},
                {"role": "user", "content": prompt},
            ]
        )
        # 获取响应内容
        sentence = chat_completion.choices[0].message.content
        return f"{sentence}"
    except Exception as e:
        return f"Error generating sentence: {e}"

def read_learned_words(filename="learned.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            learned_words = set(word.strip() for word in file.readlines())
    except FileNotFoundError:
        learned_words = set()
    return learned_words

def update_learned_words(words, filename="learned.txt"):
    with open(filename, "a", encoding="utf-8") as file:
        for word in words:
            file.write(word + "\n")

def update_sentences(sentence, filename="sentence.txt"):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(sentence + "\n\n")

def read_csv_and_generate_sentences(csv_file, learned_words):
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        words = [row for row in reader if row[0] not in learned_words]

    if len(words) < 3:
        print("Not enough new words to generate a sentence.")
        return

    selected_words = random.sample(words, 3)
    sentence = generate_sentence(*sum(selected_words, []))
    print(sentence)

    # 更新学习的单词和句子
    new_learned_words = [word[0] for word in selected_words]
    update_learned_words(new_learned_words)
    update_sentences(sentence)

# 读取已学习的单词
learned_words = read_learned_words()

# 调用函数生成句子
read_csv_and_generate_sentences('english.csv', learned_words)