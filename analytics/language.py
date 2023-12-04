# import langdetect
from ftlangdetect import detect
import re
import matplotlib.pyplot as plt

SEP = "\n----------------------------------------------------------------------------------------\n"

# with open("strings.txt") as f:
#     strings = f.read()

with open("strings_plus.txt") as f:
    strings = f.read()


strings = map(lambda x: x.strip("f").strip("\"'").strip(), strings.split(SEP))
strings = set(list(strings))
print(len(strings))

freq = {}
en = 0
ot = 0
for prompt in strings:
    prompt = re.sub("\{[^\}]*\}", "", prompt)

    if len(prompt) < 6:
        continue

    try:
        lang = detect(prompt.split("\n")[0])
        lang = lang["lang"]
    except Exception as e:
        lang = "error"

    if lang not in freq:
        freq[lang] = 0

    if lang == "en":
        en += 1
    else:
        ot += 1
    freq[lang] += 1

freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}
print(freq)
print(en)
print(ot)
print(en / ot)

langs = freq.keys()
counts = freq.values()

plt.xlabel("language")
plt.ylabel("frequency")
plt.title("Frequency of Languages in LLM Prompts")
plt.bar(langs, counts)
plt.show()
