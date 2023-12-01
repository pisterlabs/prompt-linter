# -*- coding: utf-8 -*-

import argparse
import re
import openai
from tqdm import tqdm
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize

import os
import tempfile
import shutil
import configparser

from io import StringIO
import random
import json
import chat
import pyvtt
import chardet

# 初始化翻译后的文本
text = ""
translated_text = ""

# 从文件中加载已经翻译的文本
translated_dict = {}
jsonfile = ""


with open('settings.cfg', 'rb') as f:
    content = f.read()
    encoding = chardet.detect(content)['encoding']

with open('settings.cfg', encoding=encoding) as f:
    config_text = f.read()
    config = configparser.ConfigParser()
    config.read_string(config_text)

# 获取openai_apikey和language
language_name = config.get('option', 'target-language')
lang = config.get('option', 'target-lang')


def split_text(text):
    # 使用正则表达式匹配输入文本的每个字幕块（包括空格行）
    blocks = re.split(r'(\n\s*\n)', text)

    # 初始化短文本列表
    short_text_list = []
    # 初始化当前短文本
    short_text = ""
    # 遍历字幕块列表
    for block in blocks:
        # 如果当前短文本加上新的字幕块长度不大于600，则将新的字幕块加入当前短文本
        if len(short_text + block) <= 600:
            short_text += block
        # 如果当前短文本加上新的字幕块长度大于600，则将当前短文本加入短文本列表，并重置当前短文本为新的字幕块
        else:
            short_text_list.append(short_text)
            short_text = block
    # 将最后的短文本加入短文本列表
    short_text_list.append(short_text)
    return short_text_list


def is_translation_valid(original_text, translated_text):
    def get_index_lines(text):
        lines = text.split('\n')
        index_lines = [line for line in lines if re.match(r'^\d+$', line.strip())]
        return index_lines

    original_index_lines = get_index_lines(original_text)
    translated_index_lines = get_index_lines(translated_text)

    print(original_text, original_index_lines)
    print(translated_text, translated_index_lines)

    return original_index_lines == translated_index_lines


def translate_text(text):
    global lang
    global language_name

    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            prompt = f"You are a program responsible for translating subtitles and a chess expert. Translate the following subtitle text into {language_name}, Do not combine the subtitle content of the upper and lower lines together, keep the subtitle number and timeline unchanged. \n{text}"
            print(prompt)
            response = chat.create(prompt)
            t_text = (
                response
                .get("content")
            )

            return t_text

            # if is_translation_valid(text, t_text):
            #     return t_text
            # else:
            #     retries += 1
            #     print(f"Invalid translation format. Retrying ({retries}/{max_retries})")

        except Exception as e:
            import time
            sleep_time = 60
            time.sleep(sleep_time)
            retries += 1
            print(e, f"will sleep {sleep_time} seconds, Retrying ({retries}/{max_retries})")

    print(f"Unable to get a valid translation after {max_retries} retries. Returning the original text.")
    return text


def translate_and_store(text):
    global translated_dict
    global jsonfile

    # 如果文本已经翻译过，直接返回翻译结果
    if text in translated_dict:
        return translated_dict[text]

    # 否则，调用 translate_text 函数进行翻译，并将结果存储在字典中
    translated_text = translate_text(text)
    translated_dict[text] = translated_text

    # 将字典保存为 JSON 文件
    with open(jsonfile, "w", encoding="utf-8") as f:
        json.dump(translated_dict, f, ensure_ascii=False, indent=4)

    return translated_text


def main():
    global lang
    global text
    global translated_text
    global translated_dict
    global jsonfile


    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of the input file")
    parser.add_argument("--test", help="Only translate the first 3 short texts", action="store_true")
    args = parser.parse_args()

    # 获取命令行参数
    filename = args.filename
    base_filename, file_extension = os.path.splitext(filename)
    filename_translated = base_filename + "." + lang + file_extension
    
    # 根据文件类型调用相应的函数
    if file_extension == '.srt' or file_extension == '.vtt':
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        print("Unsupported file type")

    # 从文件中加载已经翻译的文本
    translated_dict = {}
    jsonfile = base_filename + "_process.json"
    try:
        with open(jsonfile, "r", encoding="utf-8") as f:
            translated_dict = json.load(f)
    except FileNotFoundError:
        pass

    try:
        with open(jsonfile, "r", encoding="utf-8") as f:
            translated_dict = json.load(f)
    except FileNotFoundError:
        pass

    subs = pyvtt.open(filename)
    subs.clean_indexes()
    subs.save(filename, include_indexes=True)

    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # 将文本分成不大于1024字符的短文本list
    short_text_list = split_text(text)

    if args.test:
        short_text_list = short_text_list[:3]

    # 遍历短文本列表，依次翻译每个短文本
    for short_text in tqdm(short_text_list):
        # 翻译当前短文本
        translated_short_text = translate_and_store(short_text)

        # 将当前短文本和翻译后的文本加入总文本中
        translated_text += f"{translated_short_text}\n\n"
        print(translated_short_text)

    
    subs_translated = pyvtt.from_string(translated_text)
    subs_translated.clean_indexes()
    subs_translated.save(filename_translated, include_indexes=True)

    try:
        os.remove(jsonfile)
        print(f"File '{jsonfile}' has been deleted.")
    except FileNotFoundError:
        print(f"File '{jsonfile}' not found. No file was deleted.")

    return

if __name__ == "__main__":
    main()
