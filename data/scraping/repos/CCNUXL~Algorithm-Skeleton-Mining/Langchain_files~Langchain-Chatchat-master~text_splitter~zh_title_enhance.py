from langchain.docstore.document import Document
import re


def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    """Checks if the proportion of non-alpha characters in the text snippet exceeds a given
    threshold. This helps prevent text like "-----------BREAK---------" from being tagged
    as a title or narrative text. The ratio does not count spaces.

    Parameters
    ----------
    text
        The input string to test
    threshold
        If the proportion of non-alpha characters exceeds this threshold, the function
        returns False
    """
    # 这个函数的作用是检查文本中非字母字符的比例是否超过给定的阈值。
    # 它避免了类似于 "-----------BREAK---------" 这样的文本被误判为标题或叙述性文本。
    # 该比率不考虑空格。函数内部的处理步骤如下：
    # 计算文本中字母字符和总字符数（不包括空格）。
    # 然后计算字母字符数与总字符数的比值，若比值小于阈值，则返回 True，否则返回 False。
    if len(text) == 0:
        return False

    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    try:
        ratio = alpha_count / total_count
        return ratio < threshold
    except:
        return False


def is_possible_title(
        text: str,
        title_max_word_length: int = 20,
        non_alpha_threshold: float = 0.5,
) -> bool:
    """Checks to see if the text passes all of the checks for a valid title.

    Parameters
    ----------
    text
        The input text to check
    title_max_word_length
        The maximum number of words a title can contain
    non_alpha_threshold
        The minimum number of alpha characters the text needs to be considered a title
    """
    # 这个函数用于判断文本是否可能是标题。它包含了几个检查步骤：
    # 首先检查文本长度是否为零，若是，则返回False
    # 接着检查文本末尾是否为标点符号，是的话也返回 False
    # 然后检查文本长度是否超过指定的最大单词长度（默认为 20），超过则返回 False
    # 通过调用 under_non_alpha_ratio 函数来检查文本中非字母字符的比例，若超过设定的阈值，则返回 False
    # 还有一些其他检查，例如：文本是否以逗号、句号或其它标点符号结尾，或者是否全部由数字组成。最后还检查开头的字符中是否含有数字。

    # 文本长度为0的话，肯定不是title
    if len(text) == 0:
        print("Not a title. Text is empty.")
        return False

    # 文本中有标点符号，就不是title
    ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
    ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
    if ENDS_IN_PUNCT_RE.search(text) is not None:
        return False

    # 文本长度不能超过设定值，默认20
    # NOTE(robinson) - splitting on spaces here instead of word tokenizing because it
    # is less expensive and actual tokenization doesn't add much value for the length check
    if len(text) > title_max_word_length:
        return False

    # 文本中数字的占比不能太高，否则不是title
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    # NOTE(robinson) - Prevent flagging salutations like "To My Dearest Friends," as titles
    if text.endswith((",", ".", "，", "。")):
        return False

    if text.isnumeric():
        print(f"Not a title. Text is all numeric:\n\n{text}")  # type: ignore
        return False

    # 开头的字符内应该有数字，默认5个字符内
    if len(text) < 5:
        text_5 = text
    else:
        text_5 = text[:5]
    alpha_in_text_5 = sum(list(map(lambda x: x.isnumeric(), list(text_5))))
    if not alpha_in_text_5:
        return False

    return True


def zh_title_enhance(docs: Document) -> Document:
# 对一组文档进行处理，
# 如果其中有可能作为标题的文本，则在该文档的 metadata 中标记为 'cn_Title'，并修改文档内容以显示标题相关的信息。
    title = None
    if len(docs) > 0:
        for doc in docs:
            if is_possible_title(doc.page_content):
                doc.metadata['category'] = 'cn_Title'
                title = doc.page_content
            elif title:
                doc.page_content = f"下文与({title})有关。{doc.page_content}"
        return docs
    else:
        print("文件不存在")

