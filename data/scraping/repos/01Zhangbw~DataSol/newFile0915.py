import json
import os
import time
# 定义子文件夹优先级
folder_priorities = {
    "folder_0": 0,
    "folder_1": 1,
    "folder_2": 2,
    "folder_3": 3
}

# 定义当前正在处理的文件夹
current_processing_folder = None
# 定义你的 OPENAI_API_KEY
OPENAI_API_KEY = "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"

# 假设你的 chat4 函数已经定义好了
def chat4(prompt, temperature=0):
    import openai
    openai.api_key = OPENAI_API_KEY
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

    retry_count = 0  # 计数器
    while True:
        try:
            start_time = time.time()  # 记录开始时间
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=messages,
                temperature=temperature
            )
            print(f"API call completed in {time.time() - start_time} seconds.")  # 打印API调用用时
            break
        except Exception as e:
            if str(e).startswith("This model's maximum context length is"):
                raise e
            if retry_count >= 5: # 如果尝试了5次都失败，那么跳过这个消息
                print(f"Failed to process prompt after  retries: {prompt}")
                return None
            retry_count += 1  # 增加计数器
            print(f"API call failed, retrying in 1 second. ({retry_count}/5)")
            time.sleep(1)  # 增加延迟时间

    answer = completion.choices[0].message['content']
    return answer
def process_json_file(input_file):
    # 读取JSON文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 找出所有未处理的数据
    unprocessed_data = [item for item in data if item["output"] is None]

    # 逐个处理数据
    for item in unprocessed_data:
        input_prompt = item["input"]  # 获取输入 prompt

        # 调用 GPT-3.5 Turbo 模型获取输出
        output = chat4(input_prompt, temperature=0.7)

        # 更新 JSON 数据中的 output 字段
        item["output"] = output

    # 更新原始数据中的已处理数据
    for item in data:
        if item["output"] is None:
            for processed_item in unprocessed_data:
                if item["input"] == processed_item["input"]:
                    item["output"] = processed_item["output"]
                    break

    # 将更新后的数据写回文件
    with open(input_file, 'w') as f:
        json.dump(data, f, indent=4)

def process_folders(parent_folder_path):
    # 获取所有子文件夹的列表，并按优先级排序
    global current_processing_folder  # 声明为全局变量
    subfolders = sorted([f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f))],
                        key=lambda x: folder_priorities.get(x, float('inf')))

    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_folder_path, subfolder)

        # 如果当前有正在处理的文件夹且新文件夹的优先级较低，则将其放入等待队列
        if current_processing_folder and folder_priorities[subfolder] < folder_priorities[current_processing_folder]:
            print(f"File in {subfolder} placed in the waiting queue.")
            # 将文件夹添加到等待队列的逻辑（可以根据实际需求实现）

        # 如果新放入的子文件夹优先级高于正在处理的文件夹，则等待处理完毕后立即转入新放入的子文件夹
        elif current_processing_folder and folder_priorities[subfolder] > folder_priorities[current_processing_folder]:
            print(f"Waiting for processing to finish in {current_processing_folder} before processing {subfolder}.")
            # 等待处理完毕的逻辑（可以根据实际需求实现）

        # 更新当前正在处理的文件夹
        current_processing_folder = subfolder
        process_json_files_in_directory(subfolder_path)
# 处理指定目录下的所有 JSON 文件
def process_json_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            process_json_file(file_path)

# 调用函数处理所有子文件夹中的 JSON 文件
parent_folder_path = "E:\\File"
process_folders(parent_folder_path)

