from openai import OpenAI
import subprocess, os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))


def index():
    with open('../Generation/data/audio.txt', 'r') as f:
        content = f.read()

    print("Fetching GIFs...")
    summary_prompt = f"Create only 10 gif keywords or phrase that have expression from this audio script. Give me just the keywords one by one without any numbering or text before it and no quotes just plain text: {content}"
    
                # Customize prompts based on category
    #  if category == "beginner":
    #    summary_prompt = f"Create only 10 gif keywords or phrase that have expression from this audio script for beginners. Give me just the keywords one by one without any numbering or text before it and no quotes just plain text: {content}"
    # elif category == "programmer":
    #     summary_prompt = f"Create only 10 gif keywords or phrase that have expression from this audio script for programmers. Give me just the keywords one by one without any numbering or text before it and no quotes just plain text: {content}"
    # elif category == "academic":
    #     summary_prompt = f"Create only 10 gif keywords or phrase that have expression from this audio script for professors. Give me just the keywords one by one without any numbering or text before it and no quotes just plain text: {content}"
    # elif category == "funny":
    #     summary_prompt = f"Create only 10 gif keywords or phrase that have expression from this audio script using comedy. Give me just the keywords one by one without any numbering or text before it and no quotes just plain text: {content}"    else:
    #     raise ValueError("Invalid category") 
    

    response = client.chat.completions.create(
      messages=[
            {"role": "user", "content": summary_prompt}
        ],
      model="gpt-4",
    )

    with open("../Generation/data/gif_keyword.txt", "w") as f:
        f.write(response.choices[0].message.content)

    subprocess.call(["python3", "tenor.py"])
    
index()
