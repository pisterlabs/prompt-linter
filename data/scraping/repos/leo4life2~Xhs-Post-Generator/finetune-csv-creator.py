# coding=utf8

import csv
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
PROMPT = "Write some posts for teaching people how to buy Paxlovid.\n\n"

def get_data():
    all_data = []

    with open("pax-cleaned.csv", mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)

        for row in reader:
            all_data.append((row[0], row[1]))

    return all_data

def create_csv():
    # first row first col is "prompt", second col is "completion"
    with open("finetune_data.csv", mode='w', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "completion"])

        all_data = get_data()
        for title, content in all_data:
            completion = f"标题：{title}\n内容：{content}\n\n"
            writer.writerow([PROMPT, completion])

def main():
    create_csv()

if __name__ == "__main__":
    main()