import json
import os
import time
import threading

# 定义子文件夹优先级
folder_priorities = {
    "folder_0": 0,
    "folder_1": 1,
    "folder_2": 2,
    "folder_3": 3
}

# 定义你的 OPENAI_API_KEY
OPENAI_API_KEY = "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"

# 定义令牌桶类
class TokenBucket:
    def __init__(self, max_tokens, refill_rate):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill_time = time.time()

    def refill(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_refill_time
        tokens_to_add = elapsed_time * (self.refill_rate / self.max_tokens)
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill_time = current_time

    def consume(self, tokens):
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        else:
            return False

# 创建令牌桶实例
MAX_TOKENS_PER_SECOND = 4  # 限制每秒最多调用次数
REFILL_RATE = 0.25  # 令牌桶每秒自动补充令牌数
token_bucket = TokenBucket(MAX_TOKENS_PER_SECOND, REFILL_RATE)

# 假设你的 chat4 函数已经定义好了
def chat4(prompt, temperature=0):
    global token_bucket

    # 等待直到获取到足够的令牌
    while not token_bucket.consume(1):
        time.sleep(0.1)
        token_bucket.refill()

    # API 调用
    try:
        import openai
        openai.api_key = OPENAI_API_KEY

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        retry_count = 0  # 计数器
        while True:
            start_time = time.time()  # 记录开始时间
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=messages,
                temperature=temperature
            )
            print(f"API call completed in {time.time() - start_time} seconds.")  # 打印API调用用时
            break
    except Exception as e:
        print("Exception occurred:", str(e))
        if "429" in str(e):
            print("Too many requests. Waiting and retrying...")
            time.sleep(1)
            return chat4(prompt, temperature)
        raise e

    # API 调用完成后补充令牌
    token_bucket.refill()

    answer = completion.choices[0].message['content']
    return answer

# 处理JSON文件并更新数据
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

    # 将更新后的数据写回文件
    with open(input_file, 'w') as f:
        json.dump(data, f, indent=4)

# 处理指定目录下的所有 JSON 文件
def process_json_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            process_json_file(file_path)

# 处理所有子文件夹
parent_folder_path = "E:\\File"
subfolders = sorted([f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f))],
                    key=lambda x: folder_priorities.get(x, float('inf')))

for subfolder in subfolders:
    subfolder_path = os.path.join(parent_folder_path, subfolder)
    process_json_files_in_directory(subfolder_path)
