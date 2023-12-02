
from googletrans import Translator
from langdetect import detect
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
import os
import json
load_dotenv()

# read
with open("C:\@code\APIMISUSE\data\misuse_jsons\manual\merged_split_hunk_AST_filter_manual_deduplica_reduced_category.json", encoding="utf-8") as f:
    data = json.load(f)
# get commit message and change

with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_classification_v2.json', encoding="utf-8") as f:
    data_auto = f.readlines()
    data_auto = [line for line in data_auto if line != "\n"]
    print(len(data_auto))
    data_auto = [json.loads(line) for line in data_auto]

template_2 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Please fill in the infomation in the given template below.
Please give answer base on the following API misuse rules.
Please including the following points in your code review comment:
    1. Is the commit a fix of API misuse, if no, please provide the reason
    2. If it is API misuse, what is the action of the fix, if no, please say NA
    3. If it is API misuse, what is the API-element of the fix, if no, please say NA
    4. If it is API misuse, what is the motivation of the fix, if no, please say NA

How to determinte if the commit is a fix of API misuse:
    1. An API misuse is a subset of bug fix.
    2. Renaming and feature addition are not API misuse.
    2. Documents fix and update are not API misuse. 
    3. Any testing related fix are not API misuse.
    4. changes in comments are not API misuse.
    5. If the changes are only in retun value it is not API misuse.
    6. If no API used or no official API used, it is not API misuse.
    7. An API misuse fix has to be code change on API-emelemnts, such as API call, API parameter, and API condition check.
    8. The motivation of the fix has to be in the one of the following categories:
        1. Math fix: fix math error such as devide by zero.
        2. Resource fix: fix resource error such Cuda error, device problem, and CPU and GPU problem.
        3. Shape fix: fix shape error such as Tensor shape mismatch.
        4. State fix: fix state error such as state add torch.no_grad before tensor operation.
        5. Type fix: fix type error such as type mismatch.
        6. Null fix: fix null error such as checking if parameter is null before pass it to API.
        7. Argument fix: fix argument error such as argument missing.
        8. Refactor fix: fix refactor error such as update API call after refactor.
        9. Version fix: fix version error such as update API call after version update.

commit message: 
{commit_message}

code change: 
{change_code}

removed code: 
{removed_code}

added code: 
{added_code}

code change explaination:
{code_change_explaination}

template:
if_API_fix: yes or no
action_of_fix: removal, addition, change, or update
API_element_of_fix: API call, API parameter, or API condition check
motivation_of_fix: math, resource, shape, state, type, null, argument, refactor, or version
reason_of_fix: reason of the fix

"""

prompt_2 = PromptTemplate(
    input_variables=["commit_message", "change_code",
                     "removed_code", "added_code", "code_change_explaination"],
    template=template_2
)


for i in range(0, len(data_auto)):
    # for i in range(110, 114):
    print("current_index:", i, "/", len(data))

    commit_message = "{}\n".format(data[i]["commit_message"])
    change = ""
    removed = ""
    added = ""

    for j in range(0, len(data[i]["change"])):
        line = data[i]["change"][j]
        change += "{}\n".format(data[i]["change"][j])
        if line.startswith("-"):
            removed += "{}\n".format(data[i]["change"][j])
        if line.startswith("+"):
            added += "{}\n".format(data[i]["change"][j])

    code_change_explaination = data_auto[i]["code_change_explaination"]

    chatgpt_chain_2 = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt_2,
        verbose=True
    )

    output_2 = chatgpt_chain_2.predict(commit_message=commit_message, change_code=change,
                                       removed_code=removed, added_code=added, code_change_explaination=code_change_explaination)

    output_2_dict = {}
    try:
        for line in output_2.splitlines():
            if line != "":
                key, value = line.split(":")
                output_2_dict[key] = value
    except:
        pass

    output = {
        "code_change_explaination": code_change_explaination,
        "misuse_classification": output_2_dict
    }
    print(output)
    with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_classification_v2_refine.json', 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
