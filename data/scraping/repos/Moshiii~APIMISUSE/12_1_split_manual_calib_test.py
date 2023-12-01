from dotenv import load_dotenv
import os
import json
import openai
load_dotenv()

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(4))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)
input_path = "C:\@code\APIMISUSE\data\misuse_jsons\manual\merged_split_hunk_AST_filter_manual_deduplica_reduced_category.json"

manual_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\manual\manual_data_1k.json"
calib_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\calib\calib_data_1k.json"
test_1_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\test_1\\test_1_data_1k.json"
test_2_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\test_2\\test_2_data_1k.json"
# read
data=[]
with open(input_path, encoding="utf-8") as f:
    data = json.load(f)

#write to manual
with open(manual_path, "w", encoding="utf-8") as f:
    json.dump(data[:1000], f, indent=4, ensure_ascii=False)

with open(calib_path, "w", encoding="utf-8") as f:
    json.dump(data[1000:2000], f, indent=4, ensure_ascii=False)

with open(test_1_path, "w", encoding="utf-8") as f:
    json.dump(data[2000:3000], f, indent=4, ensure_ascii=False)

with open(test_2_path, "w", encoding="utf-8") as f:
    json.dump(data[3000:4000], f, indent=4, ensure_ascii=False)
