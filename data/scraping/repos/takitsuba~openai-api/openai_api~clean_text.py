import os
import sys
import openai

if __name__ == "__main__":
    redundant_text = sys.argv[1]
    direction = "次の文章を簡潔にしてください。"
    prompt = direction + "\n" + redundant_text

    openai.api_key = os.environ.get("API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    cleaned_text = response["choices"][0]["text"].strip()

    print(cleaned_text)
