import json

with open("separated-completions.json") as f:
    data = json.load(f)

with open("separated-chat-completions.json") as f:
    data.update(json.load(f))

with open("separated-langchain-tools.json") as f:
    data.update(json.load(f))

with open("separated-langchain-prompts.json") as f:
    data.update(json.load(f))

with open("separated-cohere-prompts.json") as f:
    data.update(json.load(f))


strings = []
for key, value in data.items():
    for string in value["strings"]:
        strings.append(string)

print("From llm calls", len(strings))
print("Unique from llm calls", len(set(strings)))

with open("strings.txt", "w") as f:
    f.write("\n----------\n".join(strings))

with open("separated-prompt_or_template_strings.json") as f:
    sketch_data = json.load(f)

for key, value in sketch_data.items():
    for string in value["strings"]:
        strings.append(string)

print("From llm calls + prompts", len(strings))
print("Unique from llm calls + prompts", len(set(strings)))

with open("strings_plus.txt", "w") as f:
    f.write("\n----------\n".join(strings))
