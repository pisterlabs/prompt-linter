# url = "https://learn.microsoft.com/en-us/azure/virtual-machines/disks-types"
url = "https://azure.microsoft.com/en-us/pricing/details/managed-disks/"
# url = "https://azure.microsoft.com/pricing/calculator/"
# url = "https://learn.microsoft.com/en-us/azure/virtual-machines/"

import requests
from bs4 import BeautifulSoup as bs
from markdownify import markdownify as md
from pprint import pprint
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

r = requests.get(url)
# print(r.text)
soup = bs(r.text, 'html.parser')
_t1 = md(r.text, heading_style="ATX")
import re
_t2 = re.sub(r'\n\s*\n', '\n\n', _t1)
_t3 = _t2.split("\nTable of contents\n\n")
_t4 = _t3[-1]
print(_t4)
_t5 = _t4.split("\n## Additional resources\n\n")
_t6 = _t5[0]
_t7 = _t6.split("\nTheme\n\n")
_t8 = _t7[0]
print(_t8)
# print(len(_t8))

# from langchain.text_splitter import MarkdownHeaderTextSplitter
# headers_to_split_on = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3"),
# ]
# markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# docs = markdown_splitter.split_text(t2)

# print(type(docs))

# ilen = []
# for i in docs:
#     ilen.append(len(i.page_content))
#     print(i)
#     print()

# pprint(len(docs))
# print(ilen)




# from markdownify import MarkdownConverter
# # Create shorthand method for conversion
# def md2(soup, **options):
#     return MarkdownConverter(**options).convert_soup(soup)
# t2 = md2(soup)
# # print(t2)

# class CustomMarkdownConverter(markdownify.MarkdownConverter):
#     def convert_a(self, el, text, convert_as_inline):
#         classList = el.get("class")
#         if classList and "searched_found" in classList:
#             # custom transformation
#             # unwrap child nodes of <a class="searched_found">
#             text = ""
#             for child in el.children:
#                 text += super().process_tag(child, convert_as_inline)
#             return text
#         # default transformation
#         return super().convert_a(el, text, convert_as_inline)

# # Create shorthand method for conversion
# def md4html(html, **options):
#     return CustomMarkdownConverter(**options).convert(html)

# md = md4html(r.text)
# print(md)
