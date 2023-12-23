import json


def process_strings(name, strings):
    all_trailing_whitespace = 0
    trailing_whitespace = 0
    leading_whitespace = 0
    skipped = 0
    for prompt in strings:
        if len(prompt) < 10:
            skipped += 1
            continue
        if prompt != prompt.rstrip():
            trailing_whitespace += 1
        if prompt != prompt.lstrip():
            leading_whitespace += 1
        if prompt != prompt.strip():
            all_trailing_whitespace += 1

    l = len(strings) - skipped
    print(
        f"\t{name} & {l} & {trailing_whitespace} ({trailing_whitespace / l * 100:.1f}) & {leading_whitespace} ({leading_whitespace / l * 100:.1f}) & {all_trailing_whitespace} ({all_trailing_whitespace / l * 100:.1f}) \\\\"
    )
    # print("Analyzing whitespace for:", len(strings))
    # print("Skipped (len < 10):", skipped)
    # print("Trailing whitespace:", trailing_whitespace)
    # print("Leading whitespace:", leading_whitespace)
    # print("All trailing whitespace:", all_trailing_whitespace)


print(
    """\\begin{table}
  \caption{Leading \& Trailing Whitespace Detection}
  \label{tab:whitespace}
  \\begin{tabular}{ccccc}"""
)
print("\t\\toprule")
print("\tSet & Total & Trailing (\\%) & Leading (\\%) & All (\\%) \\\\")
print("\t\\midrule")
with open("strings.json") as f:
    strings = json.load(f)
    process_strings("a, b, c, d", strings)

with open("strings_plus.json") as f:
    strings = json.load(f)
    process_strings("e", strings)

print(
    """\t\\bottomrule
\end{tabular}
\end{table}"""
)
