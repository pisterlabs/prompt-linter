import openai

openai.api_key = "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"

def gpt3(prompt, t, max_tokens):
    response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=prompt,
        temperature=t,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=""
    )
    return response["choices"][0]["text"]

with open("example_for_FQN_2.txt", "r") as f:
    prompt_examples = f.read()
tips = "# get all the Fully Qualified Names of the above simple names"
with open("result_FQN_1", "r", encoding="utf-8") as f:
    a = f.read()
    a = a.split("===================================================================")
    a = a[:len(a)-1]
    for i in range(len(a)):
        prompt = prompt_examples + a[i] + tips
        res = gpt3(prompt, 0, 256)
        result = res.split("==========")[0]
        print(a[i]+tips+result)
        print("===================================================================")
