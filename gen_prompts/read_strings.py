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
repos = set()
for repo, value in data.items():
    if value["strings"] != []:
        repos.add(repo)

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
repos_from_llm = len(repos)
strings_from_llm = len(strings)
unique_strings_from_llm = len(set(strings))
unique_strings_g_10_from_llm = len(set(filter(lambda x: len(x) >= 10, strings)))

with open("strings.json", "w") as f:
    json.dump(list(set(strings)), f, indent=2, ensure_ascii=False)

with open("separated-grouped-used_prompt_or_template_name.json") as f:
    sketch_data = json.load(f)

strings_plus = []
repos_sketch = set()
for repo, value in sketch_data.items():
    if value["strings"] != []:
        repos_sketch.add(repo)
    for string in value["strings"]:
        strings_plus.append(string)

strings_plus = list(
    map(
        lambda x: x.strip("f")
        .strip("\"'")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\r", "\r")
        .replace("\\\n", "\n")
        .replace('\\"', '"'),
        strings_plus,
    )
)

repos_from_sketch = len(repos_sketch)
strings_from_sketch = len(strings_plus)
unique_strings_from_sketch = len(set(strings_plus))
unique_strings_g_10_from_sketch = len(set(filter(lambda x: len(x) >= 10, strings_plus)))

repos_all = repos_sketch.union(repos)
strings_all = strings + strings_plus
strings_all_count = len(strings_all)
unique_strings_all = len(set(strings_all))
unique_strings_g_10_all = len(set(filter(lambda x: len(x) >= 10, strings_all)))

with open("strings_plus.json", "w") as f:
    json.dump(list(set(strings)), f, indent=2, ensure_ascii=False)

# print latex table
print(
    """\\begin{table}
  \caption{Unique Prompts}
  \label{tab:unique_prompts}
  \\begin{tabular}{ccccc}
    \\toprule
    Set & Total Found & Unique & Length > 10 & Repositories\\\\
    \\midrule"""
)

print(
    "\ta, b, c, d &",
    strings_from_llm,
    "&",
    unique_strings_from_llm,
    "&",
    unique_strings_g_10_from_llm,
    "&",
    repos_from_llm,
    "\\\\",
)
print(
    "\te &",
    strings_from_sketch,
    "&",
    unique_strings_from_sketch,
    "&",
    unique_strings_g_10_from_sketch,
    "&",
    repos_from_sketch,
    "\\\\",
)
print(
    "\ttotal &",
    strings_all_count,
    "&",
    unique_strings_all,
    "&",
    unique_strings_g_10_all,
    "&",
    len(repos_all),
    "\\\\",
)
print(
    """  \\bottomrule
\end{tabular}
\end{table}"""
)
