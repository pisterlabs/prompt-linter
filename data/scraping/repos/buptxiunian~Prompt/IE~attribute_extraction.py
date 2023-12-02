# 属性抽取
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


async def attribute_extraction(pages):
    model = "Qwen-14B-Chat-Int4"

    examples = [
        {"input": "雅生活服务（03319.HK）成立于1992年，是开发商雅居乐集团的附属公司。2017年公司收购绿地物业并引入绿地控股（600606.SH）成为其长期战略股东。2018 年 2月公司从雅居乐集团拆分后在港交所上市。公司业务涉及住宅物业服务、高端商写资产管理、公共物业服务、社区商业。按照公司收入来源划分，主营业务分为物业管理服务。",
         "output": '''{"attribute_list":[{"name":"成立时间", "content":"1992年"},{"name":"收购时间", "content":"2017年"},{"name":"上市时间", "content":"2018年2月"},{"name":"主营业务", "content":"物业管理服务"}]}'''},
        {"input": '''公司主要聚焦中高端物业，管理的资产类别有住宅物业（包括旅游地产）和非住宅物业（包括商用物业，写字楼和综合体）两大类。截止 2019H1，公司在管面积中住宅类业态占比 58.7%，非住宅类业态占比 41.3%，其中非住宅类占比增加是因为公司通过收并购项目的非住宅业态占比较高''',
         "output": '''{"attribute_list":[{"name":"住宅类业态占比", "content":"58.7%"},{"name":"非住宅类业态占比", "content":"41.3%"}]}'''}
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
            ("system", '''你现在需要完成一个**属性抽取**任务，尽可能地抽**数据信息**''' +
             '''抽取的属性请用json的形式展示，其中json的第一个元素为**属性名称**，第二个元素为**属性的具体内容**。除了这个列表以外请不要输出别的多余的话。'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=32)
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
    merged_json = {"attribute_list": []}
    with tqdm(total=len(pages)) as pbar:
        pbar.set_description('Processing:')
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
    return merged_json
