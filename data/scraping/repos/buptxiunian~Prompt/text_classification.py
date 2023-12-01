# 文本分类
import json
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
from units.merge_json import merge_json
from tqdm import tqdm
from units.load_data import load_data


async def text_classification(pages):
    model = "Qwen-14B-Chat-Int4"
    examples = [
        {"input": '''媒体盘点曾与卡扎菲家族私交甚好西方政要(图)卡扎菲独揽大权时，西方领导人不乏他的“好朋友”，包括法国总统萨科齐，英国前首相布莱尔，意大利总理贝卢斯科尼等；英法意三国均参与到利比亚军事行动中【财新网】(记者 张焕平)利比亚的独裁者卡扎菲已穷途末路，日前，国际刑警组织发出红色通告 要求逮捕卡扎菲 。
    不过，当年在卡扎菲独揽大权时，各国领导人不少是他的“好朋友”，其中不乏西方的政要，包括法国总统萨科齐，英国前首相布莱尔，意大利总理贝卢斯科尼等。
    如今，再来回顾这些政要当年与卡扎菲的亲密交往，显的有一点点突兀，但在当时是很正常的。“没有永远的敌人，没有永远的朋友”，政治是变幻的。
    财新网根据公开资料，收集了利比亚战争之前7位政要与卡扎菲家族的亲密交往。
    上述7位卡扎菲的曾经的“好朋友”，现在仅有委内瑞拉总统查韦斯、尼加拉瓜总统奥尔特加、阿尔及利亚总理乌叶海亚仍在向卡扎菲提供援手。英国、法国、意大利三国均参与到利比亚军事行动中，对昔日的“好友”开火。
    维基解密的资料显示，卡扎菲还试图与奥巴马搞好关系，卡扎菲曾写信给奥巴马，以“亲爱的美国总统奥巴马先生”为开头，祝贺其当选美国首位非洲裔总统。联合国批准对利比亚建立禁飞区之后，卡扎菲曾给奥巴马写信称其为“我们的儿子”，称无论如何奥巴马形象不会改变。
    当然，奥巴马似乎没有回应他。''',
         "output": '''{"classification_list": [{"classification": "时政"}]}'''},
        {"input": '''飓风闪达无后续资金注入解散运营团队
    新浪游戏讯，6月4日下午消息，北京飓风闪达网络科技有限公司已全面解散运营团队，据分析由于无后续资金注入资金链已断。飓风闪达内部人员日前已向新浪游戏记者确认消息属实，并表示公司运营团队已于去年年底相继离职，今年三月中旬公司正式宣布解散运营团队。
    据消息人士透露，该公司在解散运营团队过程中，与大部分合同到期未续约的员工解除合同，公司并未给予补偿；按照国家劳动法规定，职员与公司签署正式劳动合同并在其工作满一年，如果公司决定单方面解除合同，则公司要给予员工额外一个月的薪金作为补偿。
    飓风闪达旗下曾推出Q版3D mmo产品《启程-精灵的密语》，已于2008年12月31日开启不删档内测，产品在内测初期势头猛劲，但由于资方问题，后期宣传跟不上，目前服务器罕见人迹。
    据悉，飓风闪达产品在大陆内测开启之时与港台闻名游戏公司天宇科技股份有限公司(GameCyber Technology Limited )携手，以数百万的价格就《启程-精灵的密语》在港台地区的代理运营签订协议，当时韩方颇有微词。
    有业内人士透露，《启程》是根据ROSE（某款网络游戏）修改，早在04年，ROSE以种种原因，与清华同方失之交臂，未能正式进入中国大陆，但ROSE国内的私服运营得风生水起，名称为星愿。
    北京飓风闪达网络科技有限公司成立于2008年，坐落于北京市海淀区上地信息产业园区内(中关村高科技产业园区内)，是一家集网络游戏运营与研发于一体的新兴高科技公司。公司的管理层成员均从业多年，曾成功开发并运营多款业内知名网游产品。
    该公司目前拖欠‘创世奇迹’和‘龙拓’两家知名游戏广告公司广告费用未结算，广告公司在此期间也曾多次去位于国贸的东君医院投资管理公司讨要广告费用，但讨要未果。（李燕）''',
         "output": '''{"classification_list": [{"classification": "科技"}]}'''},
        {
            "input": '''澳门兔年邮票1月5日登场
    2011年1月5日，中国澳门邮政将发行《兔年》生肖邮票，与祖国内地发行兔年生肖邮票同步。此次发行的是澳门第三轮生肖邮票的第4套，全套5枚，另发行小型张1枚。
    澳门邮政发行的本轮生肖系列邮票采用天干算法推算五行属性，而明年兔的五行属性为“金兔”，所以“金兔”是这套邮票的主角。
    “金兔”属月兔类，具有一切良好人缘的性格特点：善良、热情、开朗，使与之交往的人有如沐春风的感觉，令周围的人都感到舒适愉快。除此之外，“金兔”还具有丰富的想像力，喜爱超越平凡，追求出众。
    澳门生肖邮票的发行始于1984年。第一轮(1984-1995)图案以十二生肖栩栩如生的神态展现在邮票上；第二轮(1996-2007)以简单的线条描画十二生肖头部特征，背景以花团锦簇的图案衬托；第三轮生肖邮票设计理念则以五行“金、木、水、火、土”来表达。
    (赵国胜)''',
            "output": '''{"classification_list": [{'classification': '财经'}]}'''
        }

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
            ("system", '''你现在需要完成一个**文本分类**的任务，文本类型主要包含"时政"、"科技"和"财经"三种''' +
             '''分类的结构请用json的形式展示。'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    text_splitter = CharacterTextSplitter(
        chunk_size=2048, chunk_overlap=16)
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
    merged_json = {"classification_list": []}
    with tqdm(total=len(pages)) as pbar:
        pbar.set_description('Processing:')
        for page in pages:
            texts = text_splitter.split_text(page.page_content)
            for text in texts:
                tmp = await chain.arun(input=text, return_only_outputs=True)
                try:
                    json_object = json.loads(tmp)
                    merge_json(merged_json, json_object)
                except Exception as e:
                    continue

            pbar.update(1)
    # return merged_json

    # 初始化一个空字典来存储分类和出现次数
    classification_counts = {}

    # 遍历分类列表并统计出现次数
    for item in merged_json["classification_list"]:
        classification = item["classification"]
        classification_counts[classification] = classification_counts.get(
            classification, 0) + 1

    # 找到出现次数最多的分类和次数
    most_common_classification = max(
        classification_counts, key=classification_counts.get)
    return {"classification_list": [{"classification": most_common_classification}]}
