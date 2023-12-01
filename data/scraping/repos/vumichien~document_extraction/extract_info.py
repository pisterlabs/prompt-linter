from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import os
from dotenv import load_dotenv
import camelot.io as camelot
import glob
load_dotenv()
PDF = "samples/LEXE東京East id①.pdf"
DEBUG = False
openai_api_key = os.getenv("OPENAI_API_KEY")


estate_schema = Object(
    id="estate_schema",
    description="""
        入札イベントの日時と場所。
    """,
    attributes=[
        Text(
            id="price",
            description="物件の価格"
        ),
        Text(
            id="location",
            description="物件の所在地"
        ),
        Text(
            id="structure",
            description="物件の構造"
        ),
        Number(
            id="floors",
            description="物件の階建"
        ),
        Number(
            id="floor_part",
            description="物件の階部分"
        )
    ],
    examples=[
        (
            " 価格 １，６７０万円 所在地         東京都台東区上野7丁目8-15        構造 鉄筋コンクリート造 9階建 6階部分",
            [
                {"price": "１，６７０万円", "location": "東京都荒川区荒川一丁目28-3", "structure": "鉄筋コンクリート造", "floors": 9, "floor_part": 6},
            ]
        ),
        (
            "価格  ２，５２０万円  所在地       神奈川県川崎市川崎区藤崎１－３－４  鉄骨鉄筋コンクリート造　地上15階建　8階部分",
            [
                {"price": "２，５２０万円", "location": "神奈川県川崎市川崎区藤崎１－３－４", "structure": "鉄骨鉄筋コンクリート造", "floors": 15, "floor_part": 8},
            ]
        ),
    ]
)

DICT_SCHEMA = {"price": "価格", "location": "所在地", "structure": "構造", "floors": "階建", "floor_part": "階部分"}

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",  # "gpt-3.5-turbo"
    temperature=0,
    max_tokens=4096,
    openai_api_key=openai_api_key
)


def change_key(dict_output, dict_key):
    temp_dict = dict()
    keys = list(dict_output.keys())
    for item in keys:
        if item in dict_key:
            temp_dict[dict_key[item]] = dict_output[item]
        else:
            temp_dict[item] = dict_output[item]
    return temp_dict


pdf_file = glob.glob("samples/*.pdf")
for i, pdf in enumerate(pdf_file):
    print(pdf)
    tables = camelot.read_pdf(pdf)
    text = tables[0].df.to_string()
    chain = create_extraction_chain(llm, estate_schema)
    if DEBUG:
        print(chain.prompt.format_prompt("[user_input]").to_string())
    with get_openai_callback() as cb:
        output = chain.predict_and_parse(text=text)["data"]
    response_output = change_key(output['estate_schema'][0], DICT_SCHEMA)
    print(response_output)
    print(cb)
    if i == 1:
        break
