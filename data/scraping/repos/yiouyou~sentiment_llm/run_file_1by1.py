# coding=utf-8
# python run_file_1by1.py -openaikey sk-*** -input test_notes.txt
import os
import sys
import argparse
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.callbacks import get_openai_callback


parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-openaikey', action='store', help='openai_api_key', type=str, default="")
parser.add_argument('-input', action='store', help='input text file', type=str, default="job_test.txt")
options = parser.parse_args()
os.environ["OPENAI_API_KEY"] = options.openaikey
_file = options.input


llm = OpenAI(temperature=0)
template = """
Ignore previous instructions. You are a sentiment analyst of customer comments. You assist the company in further operations by dividing customer comments into three categories: positive, negative and neutral. The main purpose is to judge whether customers have a positive attitude towards the products we are trying to sell to them. When analyzing comments, in addition to the general sentiment analysis principles, the following rules must be followed:
1) If the customer is likely to agree to our call back, it is considered positive
2) If the customer is willing to communicate further or is likely to purchase in the future, it is considered positive
3) If the main content of the comment involves numbers, phone numbers, dates, addresses or web addresses, it is considered neutral
4) If the main content of the comment is dominated by interjections, modal particles, nouns or adjectives with no obvious emotional meaning, it is considered neutral

Below are some examples of sentiment analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the sentiment classification of the comments:
{_example}

The customer comment text that requires sentiment analysis is as follows:
{_content}

Output the sentiment classification and a short reason for the classification in "classification(reason)" format, and output the analysis results in English lowercase:
"""
prompt = PromptTemplate(
    input_variables=["_content", "_example"],
    template=template,
)
chain = LLMChain(llm=llm, prompt=prompt)


def call_openai(_content, _example):
    _re = ""
    _tokens = 0
    _cost = 0
    with get_openai_callback() as cb:
        _re = chain.run(_content=_content, _example=_example)
        _tokens = cb.total_tokens
        _cost = cb.total_cost
        print(f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})")
        print(f"Cost: ${cb.total_cost}\n")
    # print(_re)
    return (_re, _tokens, _cost)


left, right = os.path.splitext(os.path.basename(_file))
_wf = f"{left}_sentiments.txt"

with open("openai_prompt.examples", "r", encoding="utf8") as ef:
    _example = "".join(ef.readlines())
# print(_example)


_seg = "-"*40
if os.path.exists(_file):
    total_cost = 0
    with open(_file, encoding='utf8') as rf:
        rf_txt = rf.readlines()
    with open(_wf, "w", encoding='utf8') as wf:
        n = 0
        for i in rf_txt:
            i_li = i.strip()
            # wf.write(f"> \"{i_li}\"\n")
            wf.write(f"{_seg}{_seg}\n")
            for j in i_li.split(". "):
                n = n + 1
                jj = ""
                if j[-1] == '.':
                    jj = j
                else:
                    jj = j+"."
                j_re = "***###***"
                j_tokens = 0
                j_cost = 0
                (j_re, j_tokens, j_cost) = call_openai(jj, _example)
                total_cost = total_cost + j_cost
                j_re = j_re.replace("\n", "")
                wf.write(f"{n}) \"{jj}\"|{j_re}\n")
            wf.write(f"{_seg}{_seg}\n\n")
        wf.write(f"\nTotal Cost: ${total_cost}\n")

