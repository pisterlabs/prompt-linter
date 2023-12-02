# 作者:lxb
import os
from typing import List, Union

from langchain.schema import Document
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser

from utils.embedding import get_embedding
from utils.pdf_utils.slice.ContentSlice import ContentSlice
from utils.pdf_utils.slice.OtherSlice import OtherSlice
from utils.pdf_utils.slice.TitleSlice import TitleSlice
from utils.pdf_utils.slice.Slice import Slice
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTText, LTChar, LTTextLine

from utils.weaviate import get_client

FONT_SIZE_TRUNCATE_DECIMAL_PLACES = 4
TABLE_THRESHOLD = 0.3


def sentence_not_end(sentence):
    return sentence.strip()[-1] not in ".。?？！!"


class FormattedPdf:
    _slice_list: List[Slice]
    _content_font_name: str
    _content_font_size: float
    _content_length: int
    _directory_structure: List[Slice]

    def __init__(self, pdf_path):
        pages = [page for page in extract_pages(pdf_path)]
        self._directory_structure = self.extract_directory_structure(pdf_path)
        self._preprocess(pages)

    def extract_directory_structure(self, pdf_path):
        with open(pdf_path, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            self._directory_structure = []
            for (level, title, dest, a, structelem) in doc.get_outlines():
                self._directory_structure.append((level, title))
        return self._directory_structure

    def _preprocess(self, pages):
        self._content_font_name, self._content_font_size, self._content_length = \
            FormattedPdf._get_content_font_and_size_and_length(pages)

        # 得到尚未合并相邻content slice的slice list，所有页的slice都集中于该list中
        self._slice_list = self._pages_to_raw_slices(pages)
        # 合并相邻的相同类型的slices
        self._merge_slices()

    def _pages_to_raw_slices(self, pages):
        slices = []
        for page in pages:
            for text_box in page:
                if isinstance(text_box, LTText):
                    text_content = text_box.get_text().strip().replace("\n", " ")
                    if text_content != "":
                        if self._is_content(text_box):
                            if not (self._is_table(text_box) or FormattedPdf._is_page_number(page, text_box)):
                                slices.append(ContentSlice(text_box))
                        elif self._is_title(text_box):
                            slices.append(TitleSlice(text_box, self._get_title_lever(text_box)))

        return slices

    @staticmethod
    def _is_page_number(page, text_box):
        page_number_position = FormattedPdf._get_page_number_position(page)
        return text_box.bbox in page_number_position

    # 定义一个函数，用于获取每一页的页码位置
    @staticmethod
    def _get_page_number_position(page):
        # 获取页面的页码
        page_number = page.pageid
        # 获取页面的布局对象列表
        layout = page.groups
        # 初始化一个空列表，用于存储页码位置
        page_number_position = []
        # 遍历布局对象列表中的每个文本框或者文本行
        for element in layout:
            if isinstance(element, LTText) or isinstance(element, LTTextLine):
                # 获取文本框或者文本行的文本内容
                text = element.get_text()
                # 如果文本内容和页码一致，将其边界框坐标添加到列表中
                if text == str(page_number):
                    page_number_position.append(element.bbox)
        # 返回页码位置列表
        return page_number_position

    # 获得全文的频率出现最高的font_name和font_size和box_length
    @staticmethod
    def _get_content_font_and_size_and_length(raw_pages):
        # 创建统计用的dict
        font_dict = {}
        font_size_dict = {}
        box_length_dict = {}
        for page in raw_pages:
            for text_box in page:
                if isinstance(text_box, LTText):
                    text = text_box.get_text().strip().replace("\n", " ")
                    if text != "":
                        box_length = round(text_box.width)
                        if box_length in box_length_dict:
                            box_length_dict[box_length] += 1
                        else:
                            box_length_dict[box_length] = 1

                        for line in text_box:
                            for char in line:
                                if isinstance(char, LTChar):
                                    char_size = round(char.size, FONT_SIZE_TRUNCATE_DECIMAL_PLACES)
                                    if char_size in font_size_dict:
                                        font_size_dict[char_size] += 1
                                    else:
                                        font_size_dict[char_size] = 1
                                    font_name = char.fontname
                                    if font_name in font_dict:
                                        font_dict[font_name] += 1
                                    else:
                                        font_dict[font_name] = 1

        most_frequent_font_size = max(font_size_dict, key=font_size_dict.get)
        most_frequent_font_name = max(font_dict, key=font_dict.get)
        most_frequent_box_length = max(box_length_dict, key=box_length_dict.get)
        return most_frequent_font_name, most_frequent_font_size, most_frequent_box_length

    # 获得当前text_box的font_name和font_size
    def _get_font_info(self, text_box):
        # self.text = text_box
        font_size_dict = {}
        font_dict = {}
        for text_line in text_box:
            for char in text_line:
                if isinstance(char, LTChar):
                    char_size = round(char.size, FONT_SIZE_TRUNCATE_DECIMAL_PLACES)
                    if char_size in font_size_dict:
                        font_size_dict[char_size] += 1
                    else:
                        font_size_dict[char_size] = 1
                    font = char.fontname
                    if font in font_dict:
                        font_dict[font] += 1
                    else:
                        font_dict[font] = 1
        most_frequent_font_size = max(font_size_dict, key=font_size_dict.get)
        most_frequent_font_name = max(font_dict, key=font_dict.get)
        return most_frequent_font_name, most_frequent_font_size

    # 判断是否是content
    def _is_content(self, text: LTText):
        font_name, font_size = self._get_font_info(text)
        return font_name == self._content_font_name and font_size == self._content_font_size

    # 如果是content，判断是否为表格
    def _is_table(self, text: LTText):
        # 获取当前LTText的box的长度
        text_box_length = text.width
        # 设置一个阈值来区分表格和正文
        # 如果当前LTText的box的长度或高度和正文内容的box的长度或高度相差很大，那么它可能是表格
        if abs(text_box_length - self._content_length) / self._content_length > TABLE_THRESHOLD:
            return True
        else:
            return False

    # 判断是否为标题
    def _is_title(self, text_box):
        # 非零、非空(非空字符串、非空列表、非空元组、非空字典、非空集合)、非None
        if self._directory_structure:
            for lever, title in self._directory_structure:
                # 将文本开头的数字去除 1. 2. 3. 去除
                # lstrip 的参数是一个字符串，表示去除开头的字符，可以是多个字符
                if text_box.get_text().strip().replace("\n", " ").lstrip("0123456789. ").lower() in title.lower():
                    return True
        else:
            font_name, font_size = self._get_font_info(text_box)
            return (font_name.lower().endswith("bold") or font_name.lower().endswith(
                "medi")) or font_size > self._content_font_size

    def _get_title_lever(self, text_box):
        for lever, title in self._directory_structure:
            if text_box.get_text().strip() in title:
                return lever

    def _merge_slices(self):
        result_list = []
        i = 0
        # current_slice = slices[0]
        while i < len(self._slice_list):
            current_slice = self._slice_list[i]
            j = i + 1
            while j < len(self._slice_list):
                if current_slice.__class__ == self._slice_list[j].__class__ and sentence_not_end(
                        current_slice.get_text()):
                    # if self.belong_to_same_child_class(current_slice, self._slice_list[j]):
                    current_slice.merge(self._slice_list[j])
                    # result_list.append(current_slice)
                else:
                    result_list.append(current_slice)
                    break
                j += 1
            if j >= len(self._slice_list):
                result_list.append(current_slice)
            i = j
        if i < len(self._slice_list):
            result_list.append(self._slice_list[i + 1:])
        self._slice_list = result_list

    def is_empty(self):
        return len(self._slice_list) < 1

    def __getitem__(self, item):
        return self._slice_list[item]

    def get_all_slices(self):
        return self._slice_list

    # 获取每一个主题的内容
    def get_topic_contents(self):
        topic_contents = []
        slice_text = ""
        for slice in self._slice_list:
            if isinstance(slice, TitleSlice):
                lever = slice.get_lever()
                # 没有目录结构，则直接按照标题进行切分
                if lever == 0:
                    if slice_text != "":
                        topic_contents.append(slice_text)
                        slice_text = ""
                else:
                    if lever <= 1 and slice_text != "":
                        print(slice.get_text())
                        topic_contents.append(slice_text)
                        slice_text = ""
            else:
                slice_text += slice.get_text()

        if slice_text != "":
            topic_contents.append(slice_text)

        return topic_contents

    # 获取langchain格式的document
    def get_langchain_documents(self):
        documents = []
        last_title = None
        for slice in self._slice_list:
            if isinstance(slice, TitleSlice):
                last_title = slice
            else:
                content = slice.get_text()
                document = Document(page_content=content,
                                    metadata={"title": last_title.get_text()
                                    if last_title is not None else None})
                documents.append(document)
        return documents

    # 将最终的list[slice]存储到weaviate数据库中
    def save_to_weaviate(self, url: str, class_name: str, property: Union[str, list[str]],
                         remove_slash_n=False):
        text_list = [slice.get_text() for slice in self._slice_list if
                     isinstance(slice, ContentSlice) or isinstance(slice, OtherSlice)]
        objects = []
        client = get_client(url)
        content = property[0] if isinstance(property, list) else property
        for text in text_list:
            obj = {
                content: text.replace("\n", "") if remove_slash_n else text,
            }
            objects.append(obj)
        vectors = get_embedding().embed_documents(text_list)
        with client.batch() as batch:
            for i, obj in enumerate(objects):
                batch.add_data_object(
                    obj,
                    class_name,
                    vector=vectors[i]
                )
