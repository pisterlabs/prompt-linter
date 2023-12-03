import json

with open("separated-prompts.json") as f:
    data = json.load(f)

with open("separated-messages.json") as f:
    data.update(json.load(f))

with open("separated-langchain.json") as f:
    data.update(json.load(f))

with open("separated-tools.json") as f:
    data.update(json.load(f))

with open("separated-cohere.json") as f:
    data.update(json.load(f))


strings = []
for key, value in data.items():
    for string in value["strings"]:
        strings.append(string)

print("From llm calls", len(strings))
print("Unique from llm calls", len(set(strings)))

with open("strings.txt", "w") as f:
    f.write("\n----------\n".join(strings))

with open("separated-name-based.json") as f:
    sketch_data = json.load(f)

for key, value in sketch_data.items():
    for string in value["strings"]:
        strings.append(string)

print("From llm calls + prompts", len(strings))
print("Unique from llm calls + prompts", len(set(strings)))

with open("strings_plus.txt", "w") as f:
    f.write("\n----------\n".join(strings))
