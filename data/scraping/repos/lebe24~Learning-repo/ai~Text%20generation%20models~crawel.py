from openai import OpenAI

client = OpenAI(
    api_key = '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',  # replace with your own API key
)

def crawler(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a webcrawler, that generate  summary of the content of the provide website  "},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# Testing the function

print(crawler("https://repack-games.com/"))