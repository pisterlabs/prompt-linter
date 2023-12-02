
from googletrans import Translator
from langdetect import detect
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
import json
load_dotenv()

# read
with open("C:\@code\APIMISUSE\data\misuse_jsons\manual\merged_split_hunk_AST_filter_manual_deduplica_reduced_category.json", encoding="utf-8") as f:
    data = json.load(f)
# get commit message and change


template_1 = """
You are an experienced software developer that can understand the meaning of code and answer questions about it.
You are designed to be able to assist with software development code reviews. 
Please provide read the following code commit and provide a code review comment.
Becareful, in the code diff with context, if line start with - means the line is removed, if line start with + means the line is added. 
only lines start with - and + are considered code changes.
Other lines are context lines, not changed.
Please including the following points in your code review comment:
    1. What is the motivation of the code change in line start with - or +.
    2. what is the potential issue if the code change in line start with - or + was not made.
    3. what is the root cause of the issue.
    4. what is the solution to the issue.


{history}
Human: {human_input}
Assistant:"""

prompt_1 = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template_1
)


template_2 = """
Assistant is an experienced software developer that can understand the meaning of code and answer questions about it.
Assistant is designed to be able to assist with software development code reviews. 
Please fill in the infomation in the given template below.
Please give answer base on the following API misuse rules and the conversation history.
Please including the following points in your code review comment:
    1. Is the commit a fix of API misuse, if no, please provide the reason
    2. If yes, what is the action of the fix
    3. If yes, what is the API-element of the fix
    4. If yes, what is the motivation of the fix


How to determinte if the commit is a fix of API misuse:
    Becareful, in the code diff with context, if line start with - means the line is removed, if line start with + means the line is added. 
    only lines start with - and + are considered code changes.
    Other lines are context lines, not changed.
    Please focus on the code changes in the code diff with context in the judgement of API misuse fix.

    1. An API misuse fix has to be a fix for a bug, not renaming or feature addition.
    2. The API has to be official machine learning API such as pytorch APi or tensorflow API, not user defined API.
    3. API misuse fix has to be on the code change line start with - or +. not context line.
    3. An API misuse fix has to be a fix on API-element, which are API call, API parameter, and API condition check such as if statement before API call.
    4. The action of the fix has to be in the one of the following categories:
        1. Removal: completly remove the API-element.
        2. Addition: add new API-element.
        3. Change: change the API-element.
        4. Update: update the API-element due to version update.
    5. The motivation of the fix has to be in the one of the following categories:
        1. Math fix: fix math error such as devide by zero.
        2. Resource fix: fix resource error such Cuda error, device problem, and CPU and GPU problem.
        3. Shape fix: fix shape error such as Tensor shape mismatch.
        4. State fix: fix state error such as state add torch.no_grad before tensor operation.
        5. Type fix: fix type error such as type mismatch.
        6. Null fix: fix null error such as checking if parameter is null before pass it to API.
        7. Argument fix: fix argument error such as argument missing.
        8. Refactor fix: fix refactor error such as update API call after refactor.
        9. Version fix: fix version error such as update API call after version update.



{history}
{human_input}

template:
if_API_fix: yes or no
action_of_fix: removal, addition, change, or update
API_element_of_fix: API call, API parameter, or API condition check
motivation_of_fix: math, resource, shape, state, type, null, argument, refactor, or version
reason_of_fix: reason of the fix

"""

prompt_2 = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template_2
)


for i in range(0, len(data)):
# for i in range(110, 114):
    print("current_index:", i,"/", len(data))
    
    query = ""
    query += "commit message: \n"
    query +=  "{}\n".format(data[i]["commit_message"])
    query += "code diff with context: \n"
    for j in range(0, len(data[i]["change"])):
        query+= "{}\n".format(data[i]["change"][j])

    # print(query)

    memory_buffer = ConversationBufferWindowMemory(k=4)

    chatgpt_chain_1 = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt_1,
        verbose=False,
        memory=memory_buffer,
    )

    output_1 = chatgpt_chain_1.predict(human_input=query)
    # print(output_1)

    chatgpt_chain_2 = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt_2,
        verbose=True,
        memory=memory_buffer,
    )

    query = ""

    output_2 = chatgpt_chain_2.predict(human_input=query)
    # print(output_2)

    output_2_dict = {}
    try:
        for line in output_2.splitlines():
            if line != "":
                key, value = line.split(":")
                output_2_dict[key] = value
    except:
        pass

    output = {
        "code_review": output_1,
        "misuse_classification": output_2_dict
    }
    print(output)
    with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_classification_v1.json', 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
