import re
import nltk
import openai
import tiktoken
from langchain.callbacks import get_openai_callback
from langdetect import detect


def stat_word_nums(text):
    """实现英文单词数量统计"""
    # 1. 将文本分词
    # 2. 统计单词数量
    # 3. 返回单词数量
    # 4. 请在下面编写代码
    # 1. 将文本分词
    words = nltk.word_tokenize(text)
    # 2. 统计单词数量
    word_nums = len(words)
    # 3. 返回单词数量
    return word_nums


def stat_para_nums(text):
    """实现英文段落数量统计"""
    # 1. 将文本分段
    # 2. 统计段落数量
    # 3. 返回段落数量
    # 4. 请在下面编写代码
    paras = text.split("\n\n")

    # 如果某个段落是以“##”开头，则认为是标题，不计入段落数量
    paras = [para for para in paras if not para.startswith("##")]
    # 2. 统计段落数量
    para_nums = len(paras)
    # 3. 返回段落数量
    return para_nums


def calc_tokens(text, model="gpt-3.5-turbo-0301"):
    """使用openai的工具，计算文本的token数量"""
    encoding = tiktoken.encoding_for_model(model)
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(text))
    return num_tokens


def track_tokens_usage(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Total tokens: {cb.total_tokens}')
        print(f'Requests: {cb.successful_requests}')

    return result


def detect_language(text):
    """检测该文本的语言，eg，若是英语则返回为‘en’"""
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    return lang


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def stat_sen_nums(text):
    """实现英文句子数量统计"""
    # 1. 将文本分句
    # 2. 统计句子数量
    # 3. 返回句子数量
    # 4. 请在下面编写代码

    # 将文本标准化

    # 1. 将文本分句
    sen_split = nltk.sent_tokenize(text)
    # 2. 统计句子数量
    sen_nums = len(sen_split)
    # 3. 返回句子数量
    return sen_nums


def read_txt(txt_path):
    # 读取txt文件
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


if __name__ == '__main__':
    text = """"""

    result = stat_sen_nums(text)

    print(calc_tokens(text))
    print(result)
