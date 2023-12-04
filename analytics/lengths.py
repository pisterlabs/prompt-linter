import re
import matplotlib.pyplot as plt

SEP = "\n----------------------------------------------------------------------------------------\n"

with open("strings.txt") as f:
    strings = f.read()

# with open("strings_plus.txt") as f:
#     strings = f.read()

strings = map(lambda x: x.strip("f").strip("\"'").strip(), strings.split(SEP))
strings = set(list(strings))
print(len(strings))

prompt_lengths = {}
for prompt in strings:
    prompt = re.sub("\{[^\}]*\}", "", prompt)

    length = min(len(prompt), 2000)
    length = length // 40 * 40

    if length not in prompt_lengths:
        prompt_lengths[length] = 0
    prompt_lengths[length] += 1

freq = {
    k: v
    for k, v in sorted(prompt_lengths.items(), key=lambda item: item[1], reverse=True)
}
print(freq)

langs = freq.keys()
counts = freq.values()

plt.xlabel("character length")
plt.ylabel("frequency")
plt.title("Distribution of Prompt Lengths")
plt.bar(langs, counts, width=20)
plt.show()
