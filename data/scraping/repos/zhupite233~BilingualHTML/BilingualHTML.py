from bs4 import BeautifulSoup as bs
import openai
import time
import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    filename="./log.txt",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(funcName)s: %(message)s",
)


def contains_only_code_tag(tag):
    """
    检查标记是否只包含一个代码标记。
    """
    code_tag = tag.find('code')
    if code_tag is None:
        return False
    code_text = code_tag.text.replace('\n', '').replace(' ', '')
    p_text = tag.text.replace('\n', '').replace(' ', '')
    if code_text == p_text:
        return True


def is_single_character_tag(tag):
    """
    判断一个标签内的文本是否只有一个字符
    只有一个字符的话，就不需要翻译了
    """
    text = tag.text.replace('\n', '').replace(' ', '')
    return len(text) == 1


def is_jump_translate(tag):
    """
    判断是否可以跳过当前标签，不用翻译。
    """
    return contains_only_code_tag(tag) or is_single_character_tag(tag)


def get_translation(prompt, code):
    """
    将 HTML 网页翻译为简体中文，返回翻译后的 HTML 代码，包裹在 <code> 标签中的文本不会被翻译。
    如果无法翻译，则返回原始的 HTML 代码。
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt} {code}"}
        ]
    )
    return completion.choices[0].message.content


def translate_tag(prompt, code):
    """
    将给定的 HTML 网页翻译为简体中文，如果翻译失败则进行重试。
    """
    max_attempts = 5
    for i in range(max_attempts):
        try:
            t_text = get_translation(prompt, code)
            time.sleep(3)
            print(t_text)
            logging.info(t_text)
            return t_text
        except Exception as e:
            sleep_time = 60
            print(e)
            logging.error(e)
            print(f"请求失败，将等待 {sleep_time} 秒后重试")
            logging.info(f"请求失败，将等待 {sleep_time} 秒后重试")
            time.sleep(sleep_time)
            print(f"开始重试第 {i + 1}/{max_attempts}")
            logging.info(f"开始重试第 {i + 1}/{max_attempts}")
    print(f"请求失败，重试次数{max_attempts}/{max_attempts}，放弃请求")
    logging.error(f"请求失败，重试次数{max_attempts}/{max_attempts}，放弃请求")


def read_api_key(path):
    with open(path, 'r') as f:
        api_key = f.read().strip()
    return api_key


def read_chatGPT_prompt(path):
    with open(path, 'r') as f:
        prompt = f.read().strip()
    return prompt


def read_page(path):
    """
    打开 HTML 网页，返回 BeautifulSoup 对象。
    """
    with open(path, 'r') as f:
        soup = bs(f, "html.parser")
    return soup


def read_json(path):
    """
    打开 JSON 文件，返回 JSON 对象。
    """
    with open(path, 'r') as f:
        json_obj = json.load(f)
    return json_obj


def write_json(path, code, mode='w'):
    """
    将 JSON 对象写入 JSON 文件。
    """
    with open(path, mode) as f:
        json.dump(code, f, ensure_ascii=False)


def write_page(path, soup):
    with open(path, 'w') as f:
        f.write(soup.prettify())


def resume_translate():
    """
    恢复之前的翻译进度
    """
    try:
        translated_texts = read_json('translated.json')
        start_index = len(translated_texts)
        print(f"从索引 {start_index + 1} 处继续翻译")
        logging.info(f"从索引 {start_index + 1} 处继续翻译")
    except:
        print("没有找到之前的翻译结果，从头开始翻译")
        logging.info("没有找到之前的翻译结果，从头开始翻译")
        start_index = 0
        translated_texts = []
    return translated_texts, start_index


def translate_page(start_time, page_num, page_count, page, prompt):
    """
    读取 HTML 页面并翻译其中的所有 <p> 标签内容为中文。
    如果翻译结果已经存在，则从上次翻译的位置继续翻译。
    翻译结果会保存在 'translated.json' 文件中。
    """
    soup = read_page(page)
    p_list = soup.find_all('p')
    count = len(p_list)
    translated_texts, start_index = resume_translate()
    page_start_time = time.time()
    for i, p in enumerate(p_list[start_index:]):
        print("↓ " * 10 + "开始翻译 " + "↓ " * 10)
        logging.info("↓ " * 10 + "开始翻译 " + "↓ " * 10)
        p_start_time = time.time()
        p_code = p.prettify()
        print(p_code)
        logging.info(p_code)
        if p_code:
            if is_jump_translate(p):
                translated_texts.append(p_code)
            else:
                translated_texts.append(translate_tag(prompt, p_code))
            write_json('translated.json', translated_texts)
        p_end_time = time.time()
        print(f"已经翻译 {i + start_index + 1}/{count}", end='，')
        logging.info(f"已经翻译 {i + start_index + 1}/{count}")
        p_elapsed_time = int(p_end_time - p_start_time)
        print(f"本段耗时 {p_elapsed_time} 秒", end='，')
        logging.info(f"本段耗时 {p_elapsed_time} 秒")
        every_p_time = int((p_end_time - page_start_time) / (i + start_index + 1))
        print(f"平均每段耗时 {every_p_time} 秒")
        logging.info(f"平均每段耗时 {every_p_time} 秒")
        page_elapsed_time = int((p_end_time - page_start_time) / 60)
        print(f"本页耗时 {page_elapsed_time} 分钟", end='，')
        logging.info(f"本页耗时 {page_elapsed_time} 分钟")
        remaining_time = int(((count - i - start_index - 1) * every_p_time) / 60)
        print(f"本页预计剩余时间 {remaining_time} 分钟")
        logging.info(f"本页预计剩余时间 {remaining_time} 分钟")
        elapsed_time = int((p_end_time - start_time) / 60)
        print(f"总耗时 {elapsed_time} 分钟", end='，')
        logging.info(f"总耗时 {elapsed_time} 分钟")
        print(f"正在翻译第 {page_num + 1}/{page_count} 页")
        logging.info(f"正在翻译第 {page_num + 1}/{page_count} 页")
        print("↑ " * 10 + "翻译完成 " + "↑ " * 10)
        logging.info("↑ " * 10 + "翻译完成 " + "↑ " * 10)
    return count


def save_translated_page(page, json):
    """
    在 HTML 文件中添加翻译后的 p 标签
    """
    translated_texts = read_json(json)
    soup = read_page(page)
    p_list = soup.find_all('p')
    for i, p in enumerate(p_list):
        text = p.prettify()
        if text:
            translated_p = bs(translated_texts[i], 'html.parser')
            p.insert_after(translated_p)
    page_cn = page.parent / (page.stem + '_cn' + page.suffix)
    write_page(page_cn, soup)


def get_translated_page(path):
    path = Path(path)
    return list(path.glob('**/*.html'))


def resume_translate_page():
    """
    恢复之前的翻译进度
    """
    try:
        translated_index = read_json('index.json')
        start_index = len(translated_index)
        print(f"从索引 {start_index + 1} 处继续翻译")
        logging.info(f"从索引 {start_index + 1} 处继续翻译")
    except:
        print("没有找到之前的翻译索引，从第一个文件开始翻译")
        logging.info("没有找到之前的翻译索引，从第一个文件开始翻译")
        start_index = 0
    return start_index


openai.api_key = read_api_key('api_key.txt')
prompt = read_chatGPT_prompt('chatGPT_prompt.txt')
path = Path('translatable')
pages = get_translated_page('translatable')
pages_count = len(list(pages))
start_time = time.time()
start_index = resume_translate_page()
for i, page in enumerate(pages[start_index:]):
    count = translate_page(start_time, i, pages_count, page, prompt)
    translated_texts = read_json('translated.json')
    if count == len(translated_texts):
        save_translated_page(page, 'translated.json')
        print(f"{page.stem} 翻译完成, 进度{i + 1}/{pages_count}")
        logging.info(f"{page.stem} 翻译完成, 进度{i + 1}/{pages_count}")
        os.remove('translated.json')
        write_json('index.json', str(page), 'a')
os.remove('index.json')
print("全部文件翻译完成")
