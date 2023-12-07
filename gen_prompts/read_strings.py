import re
import json

SEP = "\n----------------------------------------------------------------------------------------\n"

with open("separated-grouped-used_in_openai_call_sub.json") as f:
    data = json.load(f)

with open("separated-grouped-used_in_langchain_llm_call_sub.json") as f:
    data.update(json.load(f))

with open("separated-grouped-used_chat_function_sub.json") as f:
    data.update(json.load(f))

with open("separated-grouped-used_langchain_tool.json") as f:
    data.update(json.load(f))

with open("separated-grouped-used_langchain_tool_class.json") as f:
    data.update(json.load(f))

strings = []
for repo, value in data.items():
    for string in value["strings"]:
        strings.append(string)

strings = list(
    map(
        lambda x: x.strip("f")
        .strip("\"'")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\r", "\r")
        .replace("\\\n", "\n")
        .replace('\\"', '"'),
        strings,
    )
)
print("From llm calls", len(strings))
print("Unique from llm calls", len(set(strings)))

with open("strings.json", "w") as f:
    json.dump(list(set(strings)), f, indent=2, ensure_ascii=False)

with open("separated-grouped-used_prompt_or_template_name.json") as f:
    sketch_data = json.load(f)

for key, value in sketch_data.items():
    for string in value["strings"]:
        strings.append(string)

strings = list(
    map(
        lambda x: x.strip("f")
        .strip("\"'")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\r", "\r")
        .replace("\\\n", "\n")
        .replace('\\"', '"'),
        strings,
    )
)
print("From llm calls + prompts", len(strings))
print("Unique from llm calls + prompts", len(set(strings)))

with open("strings_plus.json", "w") as f:
    json.dump(list(set(strings)), f, indent=2, ensure_ascii=False)
