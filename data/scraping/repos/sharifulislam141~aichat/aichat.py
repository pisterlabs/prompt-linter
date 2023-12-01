import openai
openai.api_key = "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"

while True:
    model_engine = "text-davinci-002"
    prompt =  input("Me  :")

    completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

    message = completion.choices[0].text
    output =  "Dark Hunter 141 : " + message
     
    print(output)
