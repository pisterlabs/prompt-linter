import os
import openai

path_to_api_key=f"{os.path.expanduser('~')}/.openai/api_key"
with open(file=path_to_api_key, encoding='utf-8') as api_key:
    openai.api_key = api_key.readline().strip()

prompt = """
       Store  Region    Sales  Visitors
0    Berlin  Europe   5,000$       200
1     Paris  Europe   2,010$        10
2  New York      US  10,000$       500
3    Boston      US   2,000$        50
4   Hamburg  Europe   2,000$       500. 

average sales per region per visitor as JSON.
JSON Format:
{   "region": "average sales",
    "note": "note"
}
"""
result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=[
        {"role": "user", "content": prompt},
    ]
)
print(result.choices[0].message.content)
