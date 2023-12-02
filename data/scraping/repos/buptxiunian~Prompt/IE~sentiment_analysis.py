# 情感分析
import json
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
from units.merge_json import merge_json
from tqdm import tqdm
from units.load_data import load_data


async def sentiment_analysis(pages):
    model = "Qwen-14B-Chat-Int4"

    examples = [
        {"input": "买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质",
         "output": '''{"sentiment_list":[{"seniment": "positive"}]}'''},
        {"input": "手机不好，不喜欢，就是快递有点慢，不满意",
         "output": '''{"sentiment_list":[{"seniment": "negative"}]}'''},

    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}")
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", '''你现在需要完成一个情感分类的任务，情感的类型包含"积极"和"消极"两种''' +
             '''分类的结构请用json的形式展示'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=16)
    chain = LLMChain(
        prompt=final_prompt,
        # 温度调为0，可以保证输出的结果是确定的
        llm=ChatOpenAI(
            temperature=0,
            model_name=model,
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1")
        # output_parser=output_parser
    )
    merged_json = {"sentiment_list": []}
    with tqdm(total=len(pages)) as pbar:
        pbar.set_description(f"Processing")
        for page in pages:
            texts = text_splitter.split_text(page.page_content)
            for text in texts:
                tmp = await chain.arun(input=text, return_only_outputs=True)
                try:
                    json_object = json.loads(tmp)
                    merged_json = merge_json(merged_json, json_object)
                except Exception as e:
                    continue
            pbar.update(1)
    # 初始化一个空字典来存储分类和出现次数
    classification_counts = {}

    # 遍历分类列表并统计出现次数
    for item in merged_json["sentiment_list"]:
        classification = item["seniment"]
        classification_counts[classification] = classification_counts.get(
            classification, 0) + 1

    # 找到出现次数最多的分类和次数
    most_common_classification = max(
        classification_counts, key=classification_counts.get)
    return {"sentiment_list": [{"seniment": most_common_classification}]}
