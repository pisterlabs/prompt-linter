import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import HTMLHeaderTextSplitter


"""
텍스트를 청크단위로 나눌 수 있는 기능입니다. 
글자 단위로 자를 수 있고, 원하는 토크나이저의 맞춰서 크기로 자를 수 있습니다. 
그외 HTML과 같은 문서 또한 태그 단위로 나눌 수 있습니다.
"""
# document load
with open('../dataset/akazukin_all.txt', encoding='utf-8') as f:
    akazukin_all = f.read()


# separator로 자르고 글자 수에 맞춰서 자른다.
def char_splitter():
    text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size = 20,
        chunk_overlap  = 2,
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.split_text(akazukin_all)
    print(texts)


# "\n\n", "\n", " ", ""을 알아서 찾아서 리컬시브하게 알아서, 나눠서 글수 단위로 자른다.
def recur_char_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n\n" 따로 지정해줄 필요가 없다.
        chunk_size = 20,
        chunk_overlap  = 2,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_text(akazukin_all)
    print(texts)


# 토큰 베이스 스플릿
from transformers import GPT2TokenizerFast

def token_splitter():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=20, chunk_overlap=2)
    texts = text_splitter.split_text(akazukin_all)
    print(texts)

# HTML 스플릿
def html_splitter():
    html_string = """
    <!DOCTYPE html>
    <html>
    <body>
        <div>
            <h1>Foo</h1>
            <p>Some intro text about Foo.</p>
            <div>
                <h2>Bar main section</h2>
                <p>Some intro text about Bar.</p>
                <h3>Bar subsection 1</h3>
                <p>Some text about the first subtopic of Bar.</p>
                <h3>Bar subsection 2</h3>
                <p>Some text about the second subtopic of Bar.</p>
            </div>
            <div>
                <h2>Baz</h2>
                <p>Some text about Baz</p>
            </div>
            <br>
            <p>Some concluding text about Foo</p>
        </div>
    </body>
    </html>
    """

    # 헤드만 따고 싶다.
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on) # pip install lxml
    html_header_splits = html_splitter.split_text(html_string)
    print(html_header_splits)

if __name__=="__main__":
    char_splitter()
    recur_char_splitter()
    token_splitter()
    html_splitter()
