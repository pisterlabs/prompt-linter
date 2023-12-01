import os
import openai
import json
import re
import tiktoken
import logging
import time
import random
import pandas as pd
from dotenv import load_dotenv
from revChatGPT.V3 import Chatbot
from PyPDF2 import PdfReader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()
os.environ["http_proxy"] = "http://10.10.1.3:10000"
os.environ["https_proxy"] = "http://10.10.1.3:10000"
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openai.api_key


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


def remove_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5] ')  # 匹配中文字符的正则表达式
    text = re.sub(pattern, '', text)
    pattern = re.compile(r'[\u4e00-\u9fa5]')  # 匹配中文字符的正则表达式
    return re.sub(pattern, '', text)


# 从schema文件中获取本体（ontology - entity_label）
def get_entity_labels():
    entity_labels = []

    # 读取excel工作表6.1.xlsx - ontology
    df = pd.read_excel('./data/KGConstruction/6.1.xlsx', sheet_name='Sheet1')
    # 按行迭代数据
    for index, row in df.iterrows():
        # 读取行中的每个单元格
        entity_label = row['节点1']
        entity_labels.append(entity_label)
        entity_label = row['节点2']
        entity_labels.append(entity_label)
    
    entity_labels = list(set(entity_labels))
    entity_labels = [remove_chinese(entity_label) for entity_label in entity_labels]

    return entity_labels


# 从schema文件中获取关系（relation）
def get_relations():
    relations = []

    # 读取excel工作表6.1.xlsx - relations
    df = pd.read_excel('./data/KGConstruction/6.1.xlsx', sheet_name='Sheet1')
    # 按行迭代数据
    for index, row in df.iterrows():
        # 读取行中的每个单元格
        relation_name = row['边']
        relations.append(relation_name)
    
    relations = list(set(relations))
    relations = [remove_chinese(relation) for relation in relations]

    return relations


entity_labels = get_entity_labels()
schema_relations = get_relations()


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


def triple_extraction(paragraph: str, entity_labels: list, schema_relations: list):
    system_prompt = "I want you to act as a entity and relation extractor to help me build an academic knowledge graph from several paragraphs."
    chatbot = Chatbot(api_key=openai.api_key, system_prompt=system_prompt)
    
    prompt1 = f"""
I will give you a paragraph. Extract as many named entities as possible from it. Your answer should only contain a list and nothing else. Here is an example:
---
paragraph: 
This type of bone concentration, also present in Rincon de los Sauces (northern Patagonia), suggests that overbank facies tended to accumulate large titanosaur bones.

your answer: 
[
"bone concentration",
"northern Patagonia",
"overbank facies",
"large titanosaur bones"
]
---
Here is the paragraph you should process:
{paragraph}
"""

    # entity_list = chatbot.ask(prompt1)
    entity_list = get_completion(prompt1)
    # print(entity_list)
    
    prompt2 = "This is the entity list you have just generated: " + str(entity_list) + f"""

Classify every entity in into one of the categories in the following list. You should not classify any entity into a category that in not in the following list.

{entity_labels}

Your result should be a JSON dictionary with entities being the keys and categories being the values. There should be nothing in your answer except the JSON dictionary.
---
Here is an example:

entity list:
[
"bone concentration",
"northern Patagonia",
"overbank facies",
"large titanosaur bones"
]
your answer:
{{
"bone concentration": "Paleontology",
"northern Patagonia": "Location",
"overbank facies": "Flood plain/Overbank",
"large titanosaur bones": "Large scale lateral accretion structure"
}}
"""

    # entity_category_dict = chatbot.ask(prompt2)
    entity_category_dict = get_completion(prompt2)
    # print(entity_category_dict)
    
    prompt3 = f"""
The following is the paragraph:

{paragraph}

The following is the entity list you have just generated:

{entity_list}

Extract as many relations as possible from the paragraph. Your result should be a list of triples and nothing else. The first and third element in each triple should be in the entity list you have generated and the second element should be in the following relation category list. You should not extract any relation that is not in the following list. The relation you choose should be precise and diverse. You shouldn't use "Includes" to describe all the relations.

{schema_relations}
---
Here is an example:

paragraph: 
This type of bone concentration, also present in Rincon de los Sauces (northern Patagonia), suggests that overbank facies tended to accumulate large titanosaur bones.

entity list:
[
    "bone concentration",
    "northern Patagonia",
    "overbank facies",
    "large titanosaur bones"
]

your answer:

[
    ["bone concentration","Located","northern Patagonia"],
    ["overbank facies","accumulate","large titanosaur bones"],
]

"""

    # relation_list = chatbot.ask(prompt3)
    relation_list = get_completion(prompt3)
    # print(relation_list)
    
    try:
    
        p_entity_list = json.loads(entity_list)
        p_entity_category_dict = json.loads(entity_category_dict)
        p_relation_list = json.loads(relation_list)
        print("# JSON load successful!")
        return {
            "entity_list": p_entity_list,
            "entity_category_dict": p_entity_category_dict,
            "relation_list": p_relation_list
        }
    except:
        print("# JSON load failed!")
        return {
            "entity_list": entity_list,
            "entity_category_dict": entity_category_dict,
            "relation_list": relation_list
        }


paragraph1 = "Eight distinct facies have been defined in a 110-m thick section of the Lower Devonian Battery Point Sandstone near Gaspe, Quebec. The first is a scoured surface overlain by massive sandstone with mudstone intraclasts. Facies A and B are trough cross-bedded sandstones, with poorly-and well-defined stratification, respectively. Facies C and D consist of large isolated, and smaller multiple, sets of planar cross-stratified sandstones, respectively. Facies E comprises large sandstone-filled scours, facies F comprises ripple cross stratified fine sandstones with interbedded mudstones, and facies G comprises sets of very low angle cross-stratified sandstones. The overall context of the Battery Point Sandstone, the presence of rootlets, and the abundance of trough and planar-tabular cross bedding, all suggest a generally fluvial environment of deposition. Analysis of the facies sequence and interpretation of the primary sedimentary structures suggest that channel development began by scouring, and deposition of an intraclast lag. Above this, the two trough cross bedded facies indicate unidirectional dune migration downchannel (vector mean direction 291). The large planar tabular sets are associated with the trough cross bedded facies, but always show a large (almost 90) paleoflow divergence, suggesting lateral movement of in-channel transverse bars. The smaller planar tabular sets occur higher topographically in the fluvial system, and the rippled silts and muds indicate vertical accretion. Because of the very high ratio of in-channel sandy facies to fine-grained vertical accretion facies, and because of the evidence of lateral migration of large in-channel bars, the Battery Point River appears to resemble modern braided systems more than meandering ones."
paragraph2 = "Patagonia exhibits a particularly abundant record of Cretaceous dinosaurs with worldwide relevance. Although paleontological studies are relatively numerous, few include taphonomic information about these faunas. This contribution provides the first detailed sedimentological and taphonomical analyses of a dinosaur bone quarry from northern Neuquén Basin. At Arroyo Seco (Mendoza Province, Argentina), a large parautochthonous/autochthonous accumulation of articulated and disarticulated bones that represent several sauropod individuals has been discovered. The fossil remains, assigned to Mendozasaurus neguyelap González Riga, correspond to a large (18-27-m long) sauropod titanosaur collected in the strata of the Río Neuquén Subgroup (late Turoronian-late Coniacian). A taphonomic viewpoint recognizes a two-fold division into biostratinomic and fossil-diagenetic processes. Biostratinomic processes include (1) subaerial biodegradation of sauropod carcasses on well-drained floodplains, (2) partial or total skeletal disarticulation, (3) reorientation of bones by sporadic overbank flows, and (4) subaerial weathering. Fossil-diagenetic processes include (1) plastic deformation of bones, (2) initial permineralization with hematite, (3) fracturing and brittle deformation due to lithostatic pressure; (4) secondary permineralization with calcite in vascular canals and fractures, and (5) postfossilization bone weathering. This type of bone concentration, also present in Rincó n de los Sauces (northern Patagonia), suggests that overbank facies tended to accumulate large titanosaur bones. This taphonomic mode, referred to as ''overbank bone assemblages'', outlines the potential of crevasse splay facies as important sources of paleontological data in Cretaceous meandering fluvial systems."

entity_labels = ['Location', 'Scour surface', 'Filling structure', 'Imbricate structure', 'Thicker than the bank deposit', 'Caliche nodule', 'Approximate the depth of the riverbed', 'Freshwater lake life', 'alluvial plain', 'Fine silt and clay', 'Arcose', 'Corundum', 'Suspended load', 'General > 90°', 'Mature stage', 'Braided index (B)', 'S<1.3, B>0', 'Horizontally-laminated bed', 'Fine sandstone', 'Back swamp', 'S>2.0, B≈0', 'Natural levee', 'River flood lake', 'Large and medium water bedding', 'Old stage/Senility', 'Fluvial Facies', 'General > 2 m', 'Peat layer', 'Platinum', 'Sinuosity index (S)', 'levee', 'The thickness is not big, from ten centimeters to several meters', 'Plant roots', 'Horizontal lamination', 'Small sand grain bedding', 'Flood plain/Overbank', 'Tungsten', 'Large scale lateral accretion structure', 'Mudstone', 'Mud crack/Desiccation crack', 'Small sand lamination', 'Uranium', 'Lenticle', 'Serpentine River', '2.0>S>1.3', 'Scour structure', 'High-sinuosity river', 'Oblique bedding', '≈30m-50km', 'Sand body', 'Monazite', 'Intermittent sand grain bedding (oblique wave bedding)', 'oxbow lakes', 'Overlying sand grain bedding', 'Siltstone', 'Slabby', 'Lithic sandstone', 'Point bar', 'Gravel', 'Minerals', 'Tin', 'Low-sinuosity river', 'Graded bedding', 'Lithology', 'Distributary', 'Lag conglomerate', 'Small fan deposits', 'Wandering river/Braided river', 'Copper', 'Wormtrail', 'Small cross-bedding', 'Paleontology', 'Straight river', 'Depends on the size of the river', 'Flood fan', 'Young stage/Infancy', 'abandoned channels', 'Embankment', 'Crystal', 'Sand', 'Braided river', 'Worm boring', 'Adamas', 'Oxbow lake', 'Gravel', 'River channel', 'terraces', 'active channels', 'Horizontal bedding', 'Crosslamination', 'Breccia structure', 'Meandering river/Meandering stream', 'Sedimentary structure', 'Gold', 'Medium cross-bedding', 'Argillaceous rock', 'flood plain', 'Plant trunk']
schema_relations = ['Located', 'Thickness', 'Shape', 'Granularity', 'With', 'Width', 'Including', 'Classification', 'Dispersion of tendencies', 'Layer thickness', 'Visible', 'Identification marker', 'Sometimes can appear', 'includes', 'Always have', 'Standard', 'Exposure marker']
print("# entity labels:\n", entity_labels)
print("-" * 50)
print("# schema_relations:\n", schema_relations)
print("-" * 80)


"""
export http_proxy=10.10.1.3:10000;
export https_proxy=10.10.1.3:10000;
export no_proxy=localhost,127.0.0.1,localaddress,.localdomain.com;
"""


## 抽取新的一批河流相论文
folder_path = "./data/KGConstruction/河流相论文/"
pdf_files = []
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):  # 判断是否为 PDF 文件
        pdf_files.append(filename)
for pdf_name in pdf_files:
#     pass
# pdf_name = "8.pdf"
    print("## Processing: ", pdf_name)
    pdf_path = folder_path + pdf_name
    pdf_text = read_pdf(pdf_path)
    clean_text = pdf_text.replace("  ", " ").replace("\n", "; ").replace(';',' ')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = create_chunks(clean_text, 500, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    results = []
    # 限制同时执行的线程数为4，缓解APIConnectionError（rate_limit_exceeded）报错
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(triple_extraction, chunk, entity_labels, schema_relations): chunk for chunk in text_chunks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing chunks'):
            # 收集完成的线程处理好的结果
            response = future.result()
            if response is None:
                pass
            else:
                # 汇总关键信息抽取的结果
                results.append(response)
    wrt_path = pdf_name.replace(".pdf", ".json")
    with open('results/fluvial_facies/' + wrt_path, 'w', newline='\n') as file:
        for result in results:
            json.dump(result, file, indent=4)
            file.write('\n')  # 添加换行符


# 抽取张蕾师姐的No.1和No.2两篇论文的摘要
# result = triple_extraction(paragraph1, entity_labels, schema_relations)
# print(result)
# print(type(result))
# print(json.dumps(result, indent=4))
# with open('results/fluvial_facies/zl_no1.json', 'w', newline='\n') as file:
#     json.dump(result, file, indent=4)
# print("-" * 80)

# result = triple_extraction(paragraph2, entity_labels, schema_relations)
# print(result)
# print(type(result))
# print(json.dumps(result, indent=4))
# with open('results/fluvial_facies/zl_no2.json', 'w', newline='\n') as file:
#     json.dump(result, file, indent=4)


# 抽取王瀚老师的六篇碳酸盐岩论文
# folder_path = "./data/KGConstruction/碳酸盐岩文献标定6篇/"
# pdf_name = "Facies and climateenvironmental changes recorded on a carbonate ramp.pdf"
# pdf_path = folder_path + pdf_name
# pdf_text = read_pdf(pdf_path)
# clean_text = pdf_text.replace("  ", " ").replace("\n", "; ").replace(';',' ')
# tokenizer = tiktoken.get_encoding("cl100k_base")
# chunks = create_chunks(clean_text, 1000, tokenizer)
# text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
# results = []
# # 限制同时执行的线程数为8，缓解APIConnectionError（rate_limit_exceeded）报错
# with ThreadPoolExecutor(max_workers=8) as executor:
#     futures = {executor.submit(triple_extraction, chunk, entity_labels, schema_relations): chunk for chunk in text_chunks}
#     for future in tqdm(as_completed(futures), total=len(futures), desc='Processing chunks'):
#         # 收集完成的线程处理好的结果
#         response = future.result()
#         if response is None:
#             pass
#         else:
#             # 汇总关键信息抽取的结果
#             results.append(response)
# wrt_path = pdf_name.replace(".pdf", ".json")
# with open('results/fluvial_facies' + wrt_path, 'w', newline='\n') as file:
#     for result in results:
#         json.dump(result, file, indent=4)
#         file.write('\n')  # 添加换行符
