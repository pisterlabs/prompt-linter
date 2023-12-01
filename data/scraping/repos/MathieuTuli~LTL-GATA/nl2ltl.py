import os
import openai
import random
import pickle
import time

openai.api_key = os.getenv("OPENAI_API_KEY")


obss_raw = open("data/observations.txt").read().strip().split("\n")
obss_raw = list(map(eval, obss_raw))


formulas_raw = open("data/formulas.txt").read().strip().split("\n")
formulas_raw = list(map(eval, formulas_raw))


obss = []
formulas = []

for i in range(len(obss_raw)):
    if 'player_at_kitchen' in formulas_raw[i][0]:
        formulas_raw[i] = formulas_raw[i][1:]

    if obss_raw[i][0] == 'first':
        obss.append(obss_raw[i][1])
        formulas.append(formulas_raw[i][0])

    elif obss_raw[i][0] == 'second':
        obss.append(obss_raw[i][1])
        formulas.append(formulas_raw[i][1])

prompt_idxs = [101, 108, 466, 819, 821, 844]


def predict(test_idx):
    prompt = ""

    for i in range(len(prompt_idxs)):
        idx = prompt_idxs[i]
        prompt += str(i+1) + ". " + "NL: " + \
            obss[idx] + "\n" + "LTL: " + str(formulas[idx]) + "\n"

    prompt += str(len(prompt_idxs)+1) + ". " + "NL: " + \
        obss[test_idx] + "\n" + "LTL:"

    response = openai.Completion.create(
        model="text-ada-001",
        prompt=prompt,
        max_tokens=250,
        temperature=0,
        stop='\n'
    )
    time.sleep(1)
    return response


test_idxs = [i for i in range(len(obss_raw)) if i not in prompt_idxs and (
    "cookbook_is_examined" not in formulas[i])]

random.seed(10)
random.shuffle(test_idxs)
N = 234
absolute_correct = 0
almost_correct = 0

responses = {}
out_file = open("gpt-out.pkl", "wb")

for i in range(N):
    test_idx = test_idxs[i]

    # Remove some examples that just instruct the agent to open cookbook and eat

    resp = predict(test_idx)
    print(resp["choices"][0]["text"])
    responses[test_idx] = resp["choices"][0]["text"]

    if resp["choices"][0]["text"].strip() == formulas[test_idx]:
        absolute_correct += 1
        print("Absolute correct!")
    elif resp["choices"][0]["text"].strip().replace("(", "").replace(")", "").replace(" ", "") == formulas[test_idx].replace("(", "").replace(")", "").replace(" ", ""):
        almost_correct += 1
        print("Almost correct!")
    else:
        print("Incorrect.")

print("Absolute Accuracy:", absolute_correct / N)
print("Almost Accuracy:", (absolute_correct + almost_correct) / N)
pickle.dump(responses, out_file)
out_file.close()
