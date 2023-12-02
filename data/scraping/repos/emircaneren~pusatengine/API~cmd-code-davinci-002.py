import openai

openai.api_key = ""

while True:
    prompt = input("Tamamlama isteği : ")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.9,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    if not response["choices"]:
        print(response["message"])
    else:
        text = response["choices"][0]["text"]
        print(text)
