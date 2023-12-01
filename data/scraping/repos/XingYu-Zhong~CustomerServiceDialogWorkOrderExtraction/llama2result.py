# Using ISO-8859-1 encoding to read the file
import json
import re
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
#设置代理
import os
# os.environ['http_proxy'] = 'http://127.0.0.1:10809'
# os.environ['https_proxy'] = 'http://127.0.0.1:10809'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
def extract_keys_final(input_string):
    # 使用正则表达式查找字典的key
    keys = re.findall(r"'([^']+)':", input_string)
    return keys
def simple_progress_bar(iterable, total_length=None):
    """
    A simple progress bar to display the progress of a for loop.
    """
    total_length = total_length or len(iterable)
    for i, item in enumerate(iterable, 1):
        percentage = i / total_length * 100
        progress = int(percentage // 2)  # Divide by 2 to adjust the length of the progress bar
        print(f"\r[{'#' * progress}{'.' * (50 - progress)}] {percentage:.2f}% completed", end="")
        yield item
    print()  # Print a newline after loop completion
def construct_fake_data(system_content, user_content, assistant_content=''):
    Human = system_content+user_content
    Assistant = assistant_content
    text = f"### Human:{Human} ### Assistant:{Assistant}"
    return {
        "text": text
    }
state_file_path = "llmresult_processing_state.json"
if __name__ == '__main__':
    # Check if state file exists and get the starting index
    start_index = 0
    if os.path.exists(state_file_path):
        with open(state_file_path, 'r', encoding="UTF-8") as state_file:
            state_data = json.load(state_file)
            start_index = state_data.get("last_processed_index", 0)+1
    processed_data = []
    processed_data_file_path="session_testA_llmresult.json"
    if os.path.exists(processed_data_file_path):
        with open(processed_data_file_path, 'r', encoding="UTF-8") as data_file:
            processed_data = json.load(data_file)

    with open("session_testA_without_answer.json", "r", encoding="UTF-8") as file:
        data = json.load(file)
    chat = ChatOpenAI(model_name="ft:gpt-3.5-turbo-0613:fsrl::7tyLwcVC")
    sessions = []
    outputs = []
    # Start processing from the saved index
    for i, record in enumerate(data[start_index:], start=start_index):
        outputkeylist = extract_keys_final(str(record['output']))
        sessions.append(record['session'])
        strresult = ''
        listresult = []
        prompt_tmp = """\你是一个文本提取专家，请你根据我的要求，正确的提取内容。需求1.请你精确的提取这段对话中的{outputkey}信息。需求2.如果是卡号，身份证号，手机号码则都必须是纯数字，名字是纯中文。需求3.按照指定格式提取信息，keyvalue格式.不需要回复多余的信息，只需要提取出来的信息。请解析对应的信息:"""
        prompt = PromptTemplate.from_template(prompt_tmp)
        system_content = prompt.format(outputkey=outputkeylist)
        user_content = str(record['session'])
        messages = construct_fake_data(system_content, user_content)
        print(f'messages=>{messages}')

        chatresult = chat(messages)
        outputs.append(chatresult.content)
        print(f'chatresult=>{chatresult.content}')
        json_content = chatresult.content.replace("'", '"')
        output = json.loads(json_content)
        # Append the new processed record to the processed data
        processed_data.append({"session": record['session'],
                               "output": output})


        with open(state_file_path, 'w', encoding="UTF-8") as state_file:
            json.dump({"last_processed_index": i}, state_file)
        # Save the processed data and the last processed index
        with open(processed_data_file_path, 'w', encoding="UTF-8") as data_file:
            json.dump(processed_data, data_file, ensure_ascii=False, indent=4)




