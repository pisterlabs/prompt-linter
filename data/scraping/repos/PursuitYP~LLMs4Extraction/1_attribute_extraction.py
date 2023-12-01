""" LLMs for DeepShovel: 结构化数据抽取 - 属性值抽取 """
# 支持单篇/单段的属性值抽取，输入是pdf文件路径或者段落字符串，输出是JSON格式的结构化数据
# 需要提供抽取的属性名称列表，例如["section name",, "GPS location", "associated fossils", "lithology"]
# 以放射虫的一篇文献为例
import os
import openai
import tiktoken
from dotenv import load_dotenv
import regex as re
from tqdm import tqdm
import random
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader
import ast
import json


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


# 使用gpt-3.5-turbo抽取数据，加入异常处理机制
def extract_chunk(document, template_prompt):
    for i in range(3):  # Retry the API call up to 3 times
        try:
            prompt=template_prompt.replace('<document>', document)
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo', 
                messages=[
                    {"role": "system", "content": "I want you to act as an attribute value extractor and help me extract the values of the attributes from the following document."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return "1. " + response['choices'][0]['message']['content']
        except openai.error.RateLimitError:  # If rate limit is exceeded
            wait_time = (2 ** i) + random.random()  # Exponential backoff with jitter
            logging.warning(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
            time.sleep(wait_time)  # Wait before retrying
        except Exception as e:  # If any other error occurs
            logging.error(f"API call failed: {str(e)}")
            return None  # Return None for failure
    logging.error("Failed to call OpenAI API after multiple retries due to rate limiting.")
    return None  # Return None for failure


# 调用API并使用重试机制处理rate limit error和其他异常
def get_completion(prompt, model="gpt-3.5-turbo"):
    for i in range(3):  # Retry the API call up to 3 times
        try:
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message["content"]
        except openai.error.RateLimitError:  # If rate limit is exceeded
            wait_time = (2 ** i) + random.random()  # Exponential backoff with jitter
            logging.warning(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
            time.sleep(wait_time)  # Wait before retrying
        except Exception as e:  # If any other error occurs
            logging.error(f"API call failed: {str(e)}")
            return None  # Return None for failure
    logging.error("Failed to call OpenAI API after multiple retries due to rate limiting.")
    return None  # Return None for failure


# 使用PdfReader读取pdf文献，手动加入Page Number信息
def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text


# 传入<pdf文件路径>和带抽取的属性列表，返回抽取的结构化数据
def data_extraction_file(pdf_path, field_list, result_path):
    # 1. 执行pdf解析和切片
    pdf_text = read_pdf(pdf_path)
    clean_text = pdf_text.replace("  ", " ").replace("\n", "; ").replace(';',' ')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = create_chunks(clean_text, 1000, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

    # 2. 关键信息抽取：多线程对text_chunks处理，抽取关键信息
    question_format = "0. What is the value of the 'title' attribute"
    # 适应性地生成抽取问题，并集成到抽取提示extract_prompt中去
    for idx, field in enumerate(field_list):
        new_question = str(idx+1) + ". What is the value of the '" + field + "' attribute"
        question_format = question_format + "\n" + new_question
    document = '<document>'
    # 关键信息抽取prompt
    extract_prompt=f'''Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output \"Not specified\".
When you extract a key piece of information, include the closest page number.
---
Use the following format:
{question_format}
---
Document: \"\"\"{document}\"\"\"\n
0. What is the value of the 'title' attribute: Origin of Lower Carboniferous cherts in southern Guizhou, South China (Page 1)
1.'''
    # 多线程对text_chunks处理，抽取关键信息
    results = []
    with ThreadPoolExecutor() as executor:
        # 多线程处理
        futures = {executor.submit(extract_chunk, chunk, extract_prompt): chunk for chunk in text_chunks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing chunks'):
            # 收集完成的线程处理好的结果
            response = future.result()
            if response is not None:
                # 汇总关键信息抽取的结果
                results.append(response)
    # 进一步整理关键信息抽取结果，便于下一步格式化转换
    groups = [r.split('\n') for r in results]
    groups = [y for x in groups for y in x]
    groups = sorted(groups)
    groups = [x for x in groups if "Not specified" not in x and "__" not in x]
    zipped = groups
    # 移除太长的结果 (保留len(r) <= 180)
    zipped = [r for r in zipped if len(r) <= 180]

    # 3. 数据格式转换：根据抽取的关键信息，转换生成JSON样式的结果
    zipped_example = ["1. What is the value of the 'section name' attribute: The end-Triassic extinction event (ETE) (Page 1)", "1. What is the value of the 'section name' attribute: Katsuyama section (Page 2)", "1. What is the value of the 'section name' attribute: The Inuyama area (Page 1)", "2. What is the value of the 'location of the samples and sections' attribute: Katsuyama section, Inuyama, Japan (Page 1)", "2. What is the value of the 'location of the samples and sections' attribute: Inuyama area, central Japan (Page 2)", "2. What is the value of the 'location of the samples and sections' attribute: Rock samples from TJ-3 to TJ + 4 (3 beds above TJ + 1) continuously (Page 2)", "3. What is the value of the 'GPS location' attribute: N 35◦25.367′, E 136◦58.261 (Page 2)", "4. What is the value of the 'associated fossils' attribute: Sea surface-dwelling radiolaria (Page 1)", "4. What is the value of the 'associated fossils' attribute: Radiolarian fossils (Page 1)", "4. What is the value of the 'associated fossils' attribute: Radiolarian fossils (Page 3)", "5. What is the value of the 'lithology' attribute: Bedded chert (Page 1)", "5. What is the value of the 'lithology' attribute: Bedded chert and siliciclastic rocks (Page 2)", "5. What is the value of the 'lithology' attribute: Siliceous mudstone, bedded chert sequence, and siliciclastic rocks (Page 1)"]
    zipped_str_example = str(zipped_example)[1: -1].replace('"', "")
    field_list_example = ["section name", "location of the samples and sections", "GPS location", 
                          "associated fossils", "lithology", "number of species and genera found"]
    zipped_str = str(zipped)[1: -1].replace('"', "")
    # 数据格式转换prompt
    transform_prompt = f'''You will read a paragraph, summarise it in JSON format according to keywords and remove duplicate values.
---
Here is an example: 

PARAGRAPH
{zipped_str_example}
KEYWORDS
{field_list_example}
OUTPUT
{{
    "section name": [
        "The end-Triassic extinction event (ETE)",
        "Katsuyama section",
        "The Inuyama area"
    ],
    "location of the samples and sections": [
        "Katsuyama section, Inuyama, Japan",
        "Inuyama area, central Japan",
        "Rock samples from TJ-3 to TJ + 4 (3 beds above TJ + 1) continuously"
    ],
    "GPS location": [
        "N 35◦25.367′, E 136◦58.261"
    ],
    "associated fossils": [
        "Sea surface-dwelling radiolaria",
        "Radiolarian fossils"
    ],
    "lithology": [
        "Bedded chert",
        "Bedded chert and siliciclastic rocks",
        "Siliceous mudstone, bedded chert sequence, and siliciclastic rocks"
    ],
    "number of species and genera found": []
}}
---
Here is the paragragh you need to process, summarise it in JSON format according to keywords and remove duplicate values: 

PARAGRAPH
{zipped_str}
KEYWORDS
{field_list}
OUTPUT

'''

    response = get_completion(transform_prompt)
    res_json = ast.literal_eval(response)

    # with open(result_path, 'w', newline='\n') as file:
    #     json.dump(res_json, file, indent=4)

    return res_json


# 传入<段落字符串>和带抽取的属性列表，返回抽取的结构化数据
def data_extraction_paragraph(paragraph, field_list, result_path):
    # 1. 关键信息抽取：多线程对text_chunks处理，抽取关键信息
    question_format = "0. What is the value of the 'title' attribute"
    # 适应性地生成抽取问题，并集成到抽取提示extract_prompt中去
    for idx, field in enumerate(field_list):
        new_question = str(idx+1) + ". What is the value of the '" + field + "' attribute"
        question_format = question_format + "\n" + new_question
    document = '<document>'
    # 关键信息抽取prompt
    extract_prompt=f'''Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output \"Not specified\".
When you extract a key piece of information, include the closest page number.
---
Use the following format:
{question_format}
---
Document: \"\"\"{document}\"\"\"\n
0. What is the value of the 'title' attribute: Origin of Lower Carboniferous cherts in southern Guizhou, South China (Page 1)
1.'''
    # 对paragraph处理，抽取关键信息
    results = []
    response = extract_chunk(paragraph, extract_prompt)
    results.append(response)
    # 进一步整理关键信息抽取结果，便于下一步格式化转换
    groups = [r.split('\n') for r in results]
    groups = [y for x in groups for y in x]
    groups = sorted(groups)
    groups = [x for x in groups if "Not specified" not in x and "__" not in x]
    zipped = groups
    # 移除太长的结果 (保留len(r) <= 180)
    zipped = [r for r in zipped if len(r) <= 180]

    # 2. 数据格式转换：根据抽取的关键信息，转换生成JSON样式的结果
    zipped_example = ["1. What is the value of the 'section name' attribute: The end-Triassic extinction event (ETE) (Page 1)", "1. What is the value of the 'section name' attribute: Katsuyama section (Page 2)", "1. What is the value of the 'section name' attribute: The Inuyama area (Page 1)", "2. What is the value of the 'location of the samples and sections' attribute: Katsuyama section, Inuyama, Japan (Page 1)", "2. What is the value of the 'location of the samples and sections' attribute: Inuyama area, central Japan (Page 2)", "2. What is the value of the 'location of the samples and sections' attribute: Rock samples from TJ-3 to TJ + 4 (3 beds above TJ + 1) continuously (Page 2)", "3. What is the value of the 'GPS location' attribute: N 35◦25.367′, E 136◦58.261 (Page 2)", "4. What is the value of the 'associated fossils' attribute: Sea surface-dwelling radiolaria (Page 1)", "4. What is the value of the 'associated fossils' attribute: Radiolarian fossils (Page 1)", "4. What is the value of the 'associated fossils' attribute: Radiolarian fossils (Page 3)", "5. What is the value of the 'lithology' attribute: Bedded chert (Page 1)", "5. What is the value of the 'lithology' attribute: Bedded chert and siliciclastic rocks (Page 2)", "5. What is the value of the 'lithology' attribute: Siliceous mudstone, bedded chert sequence, and siliciclastic rocks (Page 1)"]
    zipped_str_example = str(zipped_example)[1: -1].replace('"', "")
    field_list_example = ["section name", "location of the samples and sections", "GPS location", 
                          "associated fossils", "lithology", "number of species and genera found"]
    zipped_str = str(zipped)[1: -1].replace('"', "")
    # 数据格式转换prompt
    transform_prompt = f'''You will read a paragraph, summarise it in JSON format according to keywords and remove duplicate values.
---
Here is an example: 

PARAGRAPH
{zipped_str_example}
KEYWORDS
{field_list_example}
OUTPUT
{{
    "section name": [
        "The end-Triassic extinction event (ETE)",
        "Katsuyama section",
        "The Inuyama area"
    ],
    "location of the samples and sections": [
        "Katsuyama section, Inuyama, Japan",
        "Inuyama area, central Japan",
        "Rock samples from TJ-3 to TJ + 4 (3 beds above TJ + 1) continuously"
    ],
    "GPS location": [
        "N 35◦25.367′, E 136◦58.261"
    ],
    "associated fossils": [
        "Sea surface-dwelling radiolaria",
        "Radiolarian fossils"
    ],
    "lithology": [
        "Bedded chert",
        "Bedded chert and siliciclastic rocks",
        "Siliceous mudstone, bedded chert sequence, and siliciclastic rocks"
    ],
    "number of species and genera found": []
}}
---
Here is the paragragh you need to process, summarise it in JSON format according to keywords and remove duplicate values: 

PARAGRAPH
{zipped_str}
KEYWORDS
{field_list}
OUTPUT

'''

    response = get_completion(transform_prompt)
    res_json = ast.literal_eval(response)

    # with open(result_path, 'w', newline='\n') as file:
    #     json.dump(res_json, file, indent=4)

    return res_json


if __name__ == '__main__':
    # 环境初始化，用户上传OpenAI API key
    load_dotenv()
    os.environ["http_proxy"] = "http://10.10.1.3:10000"
    os.environ["https_proxy"] = "http://10.10.1.3:10000"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY'] = openai.api_key


    # LLMs结构化数据抽取 - 属性值抽取 - 输入是一个 <文件> （单篇）
    pdf_path = "../data/radiolarian/715.pdf"
    field_list = ["section name", "location of the samples and sections", "GPS location", 
                  "associated fossils", "lithology", "number of species and genera found"]
    result_path = "../results/result_1.json"
    res_json = data_extraction_file(pdf_path, field_list, result_path)
    print(json.dumps(res_json, indent=4))
    print("-" * 100)


    # LLMs结构化数据抽取 - 属性值抽取 - 输入是一个 <段落> （单段）
    ## 取"GEOLOGICAL SETTING"和"MATERIALS AND METHODS"之间的文本作为示例段落
    pdf_path = "../data/radiolarian/715.pdf"
    pdf_text = read_pdf(pdf_path)
    clean_text = pdf_text.replace("  ", " ").replace("\n", "; ").replace(';',' ')
    # re过滤clean_text中位于"GEOLOGICAL SETTING"和"MATERIALS AND METHODS"之间的文本
    pattern = re.compile(r'GEOLOGICAL SETTING AND LITHOSTRATIGRAPHY OF THE STUDY SECTION(.*?)MATERIALS AND METHODS', re.S)
    clean_text = re.findall(pattern, clean_text)[0]
    # re过滤删除clean_text字符串前后的空格
    pattern = re.compile(r'^\s*(.*?)\s*$', re.S)
    clean_text = re.findall(pattern, clean_text)[0]

    paragraph = clean_text
    field_list = ["section name", "location of the samples and sections", "GPS location", 
                  "associated fossils", "lithology", "number of species and genera found"]
    result_path = "../results/result_2.json"
    res_json = data_extraction_paragraph(paragraph, field_list, result_path)
    print(json.dumps(res_json, indent=4))
