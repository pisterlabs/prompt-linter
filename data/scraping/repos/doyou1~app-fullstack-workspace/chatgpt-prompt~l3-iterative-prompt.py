import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from IPython.display import display, HTML

openai.api_key = os.getenv("OPENAI_API_KEY")

# helper function
# Throughout this course, we will use OpenAI's gpt-3.5-turbo model and the chat completions endpoint.
# This helper function will make it easier to use prompts and look at the generated outputs.
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# Generate a marketing product description from a product fact sheet

# (translation)
# 개요
# - 세기 중반에 영감을 받은 사무실 가구의 아름다운 가족의 일부,
# 캐비닛, 책상, 책장, 회의 테이블 등을 채우는 것을 포함하여.
# - 셸 색상 및 베이스 마감의 여러 옵션.
# - 플라스틱 후면 및 전면 커버와 함께 사용 가능(SWC-100)
# 또는 10개의 패브릭 및 6개의 가죽 옵션으로 전체 커버(SWC-110)를 사용할 수 있습니다.
# - 기본 마감 옵션은 스테인레스 스틸, 무광 블랙, 광택 화이트 또는 크롬입니다.
# - 의자는 팔걸이 유무에 관계없이 사용할 수 있습니다.
# - 가정 또는 기업 환경에 적합합니다.
# - 계약 사용 자격이 있습니다.

# 시공
# - 5륜 플라스틱 코팅 알루미늄 베이스.
# - 공기식 의자 조정으로 상승/하강이 용이합니다.

# 치수
# - 폭 53CM | 20.87"
# - 깊이 51CM | 20.08"
# - 높이 80CM | 31.50"
# - 시트 높이 44CM | 17.32"
# - 시트 깊이 41CM | 16.14"

# 옵션들
# - 소프트 플로어 또는 하드 플로어 캐스터 옵션.
# - 시트 폼 밀도의 두 가지 선택:
#  중간(1.8lb/ft3) 또는 높음(2.8lb/ft3)
# - 암리스 또는 8위치 PU 암레스트

# 자재
# 셸 베이스 글라이더
# - 변형된 나일론 PA6/PA66 코팅이 적용된 주조 알루미늄.
# - 셸 두께: 10mm.
# 좌석.
# - HD36 폼

# 원산지
# - 이탈리아
fact_sheet_chair = """
OVERVIEW
- Part of a beautiful family of mid-century inspired office furniture,
including filling cabinets, desks, bookcases, meeting tables, and more.
- Several options of shell color and base finishes.
- Available with plastic back and front upholstery (SWC-100)
or full upholstery (SWC-110) in 10 fabric and 6 leather options.
- Base finish options are: stainless steel, matte black, gloss white, or chrome.
- Chair is available with or without armrests.
- Suitable for home or business settings.
- Qualified for contract use.

CONSTRUCTION
- 5-wheel plastic coated aluminum base.
- Pneumatic chair adjust for easy raise/lower action.

DIMENSIONS
- WIDTH 53 CM | 20.87”
- DEPTH 51 CM | 20.08”
- HEIGHT 80 CM | 31.50”
- SEAT HEIGHT 44 CM | 17.32”
- SEAT DEPTH 41 CM | 16.14”

OPTIONS
- Soft or hard-floor caster options.
- Two choices of seat foam densities:
 medium (1.8 lb/ft3) or high (2.8 lb/ft3)
- Armless or 8 position PU armrests

MATERIALS
SHELL BASE GLIDER
- Cast Aluminum with modified nylon PA6/PA66 coating.
- Shell thickness: 10 mm.
SEAT
- HD36 foam

COUNTRY OF ORIGIN
- Italy
"""

# (translation)
# 당신의 임무는 마케팅 팀이 다음을 만들도록 돕는 것입니다 
# 기술 자료 시트를 기반으로 한 제품의 소매 웹 사이트에 대한 설명.

# 정보를 기반으로 제품 설명을 작성 
# 다음으로 구분된 기술 사양에 제공됩니다 
# 트리플 백틱

# 기술 사양: '''{fact_sheet_chair}'''
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

Technical specifications: ```{fact_sheet_chair}```
"""
# response = get_completion(prompt)
# print(response)

# (translation)
# 20세기 중반에 영감을 받은 멋진 사무용 의자인 SWC-100과 SWC-110을 소개합니다. 아름다운 사무용 가구 제품군의 일부인 이 의자는 장식에 완벽하게 어울리는 몇 가지 쉘 색상과 베이스 마감 옵션으로 사용할 수 있습니다. 10가지 패브릭 및 6가지 가죽 옵션의 플라스틱 뒷면 및 앞면 덮개 또는 전체 덮개 중에서 선택할 수 있습니다. 기본 마감 옵션에는 스테인리스 스틸, 무광 블랙, 광택 화이트 또는 크롬이 포함됩니다.

# 이 의자는 가정과 업무용으로 설계되었으며 계약 용도로 적합합니다. 5륜 플라스틱 코팅 알루미늄 베이스와 공압식 의자 조정으로 의자를 원하는 높이로 쉽게 올리고 내릴 수 있습니다.

# 의자의 치수는 가로 53cm, 세로 51cm, 높이 80cm이며, 좌석 높이는 44cm, 좌석 깊이는 41cm입니다. 선택할 수도 있습니다 
# 부드러운 바닥 또는 단단한 바닥 캐스터 옵션과 중간(1.8lb/ft3) 또는 높은(2.8lb/ft3)의 두 가지 시트 폼 밀도 사이.

# SWC-100 및 SWC-110은 팔걸이 유무에 관계없이 8단계 PU 팔걸이 옵션을 사용할 수 있습니다. 이 의자의 구조에 사용되는 재료는 최상의 품질로, 나일론 PA6/PA66 코팅이 변경된 주조 알루미늄 쉘과 10mm의 쉘 두께를 가지고 있습니다. 이 시트는 궁극의 편안함을 위해 HD36 폼으로 제작되었습니다.

# 이 의자는 이탈리아에서 만들어졌고 어떤 사무실 공간에도 완벽하게 추가됩니다. 지금 바로 SWC-100 및 SWC-110 사무용 의자로 작업 공간을 업그레이드하십시오.

# Introducing our stunning mid-century inspired office chair, the SWC-100 and SWC-110. Part of a beautiful family of office furniture, this chair is available in several shell color and base finish options to perfectly match your decor. Choose from plastic back and front upholstery or full upholstery in 10 fabric and 6 leather options. The base finish options include stainless steel, matte black, gloss white, or chrome.

# This chair is designed for both home and business settings and is qualified for contract use. The 5-wheel plastic coated aluminum base and pneumatic chair adjust make it easy to raise and lower the chair to your desired height.

# The dimensions of the chair are 53 cm in width, 51 cm in depth, and 80 cm in height, with a seat height of 44 cm and seat depth of 41 cm. You can also choose 
# between soft or hard-floor caster options and two choices of seat foam densities: medium (1.8 lb/ft3) or high (2.8 lb/ft3).

# The SWC-100 and SWC-110 are available with or without armrests, with the option of 8 position PU armrests. The materials used in the construction of this chair are of the highest quality, with a cast aluminum shell with modified nylon PA6/PA66 coating and a shell thickness of 10 mm. The seat is made of HD36 foam for ultimate comfort.

# This chair is made in Italy and is the perfect addition to any office space. Upgrade your workspace with the SWC-100 and SWC-110 office chair today.


# Issue 1: The text is too long
 # Limit the number of words/sentences/characters.

prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
# response = get_completion(prompt)
# print(response)

# (translation)
# 아름다운 가구 제품군의 일부인 세기 중반에 영감을 받은 사무용 의자를 소개합니다. 
# 다양한 쉘 색상과 베이스 마감, 플라스틱 또는 
# 패브릭 또는 가죽 소재의 전체 실내 장식 옵션. 
# 가정용 또는 비즈니스용으로 적합하며, 5륜 베이스 및 공압식 의자 조정 기능이 있습니다. 메이드 인 이탈리아산.

# Introducing the mid-century inspired office chair, part of a beautiful furniture family. 
# Available in various shell colors and base finishes, with plastic or 
# full upholstery options in fabric or leather. 
# Suitable for home or business use, with a 5-wheel base and pneumatic chair adjust. Made in Italy.

# Issue 2. Text focuses on the wrong details
 # Ask it to focus on the aspects that are relevant to the intended audience.

prompt = f"""
Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.

Write a product description based on the information provided in the technical specifications delimited by triple backticks.

The description is intended for furniture retailers, so should be technical in nature and focus on the materials the product is constructed from.

At the end of the description, include every 7-character Product ID in the technical specification.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
# response = get_completion(prompt)
# print(response)

# (translation)
# 가정이나 비즈니스 환경에 적합한 세기 중반의 영감을 받은 사무용 의자를 소개합니다. 
# 다양한 쉘 색상과 베이스 마감, 다양한 직물과 가죽 소재의 플라스틱 또는 풀 커버 옵션을 갖춘 이 의자는 스타일리시하면서도 다재다능률적입니다. 
# 5륜 플라스틱 코팅 알루미늄 베이스와 공압식 의자 조절 장치로 구성되어 있으며, 편안하고 사용하기 쉽습니다. 모든 7자 상품 ID: SWC-100, SWC-110.
# Introducing our mid-century inspired office chair, perfect for home or business settings. 
# With a range of shell colors and base finishes, and the option of plastic or full upholstery in various fabrics and leathers, this chair is both stylish and versatile. 
# Constructed with a 5-wheel plastic coated aluminum base and pneumatic chair adjust, it's also comfortable and easy to use. Every 7-character Product ID: SWC-100, SWC-110.

# (translation)
# 3호. 설명에 치수 표가 필요합니다
 # 정보를 추출하여 표에 정리하도록 요청합니다.
# Issue 3. Description needs a table of dimensions
 # Ask it to extract information and organize it in a table.

prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

At the end of the description, include every 7-character 
Product ID in the technical specification.

After the description, include a table that gives the 
product's dimensions. The table should have two columns. 
In the first column include the name of the dimension. 
In the second column include the measurements in inches only. 

Give the table the title 'Product Dimensions'.

Format everything as HTML that can be used in a website.
Place the description in a <div> element.

Technical specifications: ```{fact_sheet_chair}```
"""

response = get_completion(prompt)
print(response)
# <div>
# <p>The mid-century inspired office chair is a perfect addition to any home or business setting. The chair is part of a beautiful family of office furniture that includes filling cabinets, desks, bookcases, meeting tables, and more. The chair is available in several options of shell color and base finishes, allowing you to customize it to your liking. You can choose between plastic back and front upholstery or full upholstery in 10 fabric and 6 leather options. The chair is also available with or without armrests. The base finish options are stainless steel, matte black, gloss white, or chrome. The chair is constructed with a 5-wheel plastic coated aluminum base and has a pneumatic chair adjust for easy raise/lower action. The chair is qualified for contract use and is made in Italy.</p>

# <h2>Product Dimensions</h2>
# <table>
#   <tr>
#     <td>Width</td>
#     <td>20.87 inches</td>
#   </tr>
#   <tr>
#     <td>Depth</td>
#     <td>20.08 inches</td>
#   </tr>
#   <tr>
#     <td>Height</td>
#     <td>31.50 inches</td>
#   </tr>
#   <tr>
#     <td>Seat Height</td>
#     <td>17.32 inches</td>
#   </tr>
#   <tr>
#     <td>Seat Depth</td>
#     <td>16.14 inches</td>
#   </tr>
# </table>
# </div>

# Product IDs: SWC-100, SWC-110.

# Load Python libraries to view HTML
display(HTML(response))