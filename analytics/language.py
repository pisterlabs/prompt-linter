import json

# import langdetect
from ftlangdetect import detect
import re
import matplotlib.pyplot as plt

# with open("strings.json") as f:
#     strings = json.load(f)

with open("strings_plus.json") as f:
    strings = json.load(f)

print(len(strings))

freq = {}
en = 0
ens = []
for prompt in strings:
    prompt_l = re.sub("\{[^\}]*\}", "", prompt)

    if len(prompt) < 6:
        continue

    try:
        lang = detect(prompt_l.split("\n")[0])
        lang = lang["lang"]
    except Exception as e:
        lang = "error"

    if lang not in freq:
        freq[lang] = 0

    if lang == "en":
        en += 1
        ens.append(prompt)
    freq[lang] += 1

freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}
print(freq)

langs = freq.keys()
counts = freq.values()

with open("strings_en.json", "w") as f:
    json.dump(ens, f, indent=2, ensure_ascii=False)

plt.xlabel("language")
plt.ylabel("frequency")
plt.title("Frequency of Languages in LLM Prompts")
plt.bar(langs, counts)
plt.show()
