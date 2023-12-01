from langchain.tools import tool
from langchain.tools import StructuredTool

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import get_openai_callback

from repolya._const import WORKSPACE_TOOLSET
from repolya._log import logger_toolset
from repolya.rag.vdb_faiss import (
    get_faiss_OpenAI,
    get_faiss_HuggingFace,
)
from repolya.rag.digest_urls import urls_to_faiss_OpenAI
from repolya.rag.qa_chain import qa_vdb_multi_query
from repolya.toolset.tool_latent import ACTIVE_LATENT_TEMPLATE, ACTIVE_LATENT_TEMPLATE_ZH

from unittest.mock import patch
from openai.error import RateLimitError
import time
import re
import os



_bp_10 = {
    "Company Purpose": "Define your company in a single declarative sentence.",
    "Problem": "Describe the pain of your customer (or the customer's customer). Outline how the customer addresses the issue today and what are the shortcomings of current solutions.",
    "Solution": "Why is your value prop unique and compelling? Why will it endure? Provide use cases.",
    "Why Now": "Set up the historical evolution of your category. Why hasn't your solution been built before now? Define recent trends that make your solution possible.",
    "Market Potential": "Identify your customer and your market. Calculate the TAM (top-down), SAM (bottoms-up), and SOM.",
    "Competition/Alternatives": "Who are your direct and indirect competitors? List competitive advantages. Show that you have a plan to win.",
    "Product": "Product line-up (form factors, functionality, features, packaging, IP). Development roadmap.",
    "Business Model": "How do you intend to thrive? Revenue model, sales&distribution model, pricing, customer list, etc.",
    "Team": "Tell the story of your founders and key team members. Board of Directors/Advisors.",
    "Financials": "If you have any, please include (P&L, Balance sheet, Cash flow, Cap table, The deal, etc.)",
    "Vision": "If all goes well, what will you have built in five years?",
}

_bp_10_zh = {
    "公司宗旨": "用一个宣告性的句子定义您的公司。",
    "市场痛点": "描述您的客户（或客户的客户）的痛点。概述客户如何处理这个问题，以及当前解决方案的不足之处。",
    "解决方案": "为什么您的价值主张是独特和引人注目的？为什么它会持久？请提供使用案例。",
    "时机": "概述公司类别的历史演变。为什么以前没有您的解决方案？最近哪些趋势使您的解决方案成为可能？",
    "市场空间": "确定您的客户和市场。计算TAM（自上而下），SAM（自下而上）和SOM。",
    "竞争态势": "谁是您的直接和间接竞争对手？列出他们的竞争优势。论述您的获胜计划。",
    "产品": "产品线（形态、功能、特点、包装、知识产权）。开发路线图。",
    "商业模式": "您打算如何繁荣发展？概述收入模型、销售和分销模型、定价、潜在客户等。",
    "团队": "讲述您的创始人和关键团队成员的故事。董事会/顾问。",
    "财务预测": "如果有的话，请附上损益表、资产负债表、现金流量表、股本结构、重要交易等。",
    "愿景": "如果一切顺利，五年后您将建立什么？",
}


# _sys = ACTIVE_LATENT_TEMPLATE_ZH.replace('<<QUERY>>', '请问可以提出哪些问题和角度？')
_sys = ACTIVE_LATENT_TEMPLATE.replace('<<QUERY>>', 'What questions and angles can you ask?')


def get_inspiration(_category, _topic):
    _human = f"""假定您正在撰写一份关于{_category}的商业BP，需要从已构建的与{_category}有关的商业案例数据库中寻找灵感，并完善自己的BP。
    
下面是在寻找"商业模式"的相关灵感时，值得思考一些问题和角度：
1. 商业模式构成：新式茶饮的核心商业模式是什么？它是如何创造价值的？这包括了哪些关键资源、关键活动和关键合作伙伴？
2. 客户群体：新式茶饮业目标的主要消费群体是哪些？他们的消费习惯、喜好和消费力如何？
3. 价值主张：新式茶饮业的独特价值主张是什么？它如何满足客户的需求并区别于竞品？
4. 销售和营销策略：新式茶饮店如何吸引和保留客户？采用了哪些有效的推广策略？
5. 收入来源：新式茶饮店的主要收入来源是什么？单一产品的定价策略如何？有无其他利润增长点（如附加值服务或产品）？
6. 成本结构：核心的成本支出是什么？如何通过效率提升或其他方式控制成本？
7. 供应链管理：新式茶饮业的供应链是怎样的？优质的原材料是如何采购的？物流和储存不同茶饮配料的方式又是如何？
8. 创新元素：新式茶饮业有哪些创新的影响力或有潜力的商业模式？这些创新如何推动企业的持续增长？
9. 竞争环境：在新式茶饮行业中，主要的竞争对手是哪些？他们的成功因素或失败因素是什么？有无可能出现新的竞争者或潜在威胁？
10. 商业可持续性：新式茶饮业对环境和社会的影响如何进行管理和减少？有无其他可持续发展的商业做法或潜在机会？

为了更全面地思考和完善新式茶饮商业BP的{_topic}部分，请问可以提出哪些问题和角度？
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _sys),
            ("human", _human)
        ]
    )
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    runnable = (
        {"_category": RunnablePassthrough(), "_topic": RunnablePassthrough()} 
        | prompt 
        | model 
        | StrOutputParser()
    )
    with get_openai_callback() as cb:
        _re = runnable.invoke({"_category": _category, "_topic": _topic})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        return _re, _token_cost

# 在寻找灵感以完善商业模式时，可以考虑以下一些问题和角度：
# 1. 商业模式构成：新式茶饮的核心商业模式是什么？它是如何创造价值的？这包括了哪些关键资源、关键活动和关键合作伙伴？
# 2. 客户群体：新式茶饮业目标的主要消费群体是哪些？他们的消费习惯、喜好和消费力如何？
# 3. 价值主张：新式茶饮业的独特价值主张是什么？它如何满足客户的需求并区别于竞品？
# 4. 销售和营销策略：新式茶饮店如何吸引和保留客户？采用了哪些有效的推广策略？
# 5. 收入来源：新式茶饮店的主要收入来源是什么？单一产品的定价策略如何？有无其他利润增长点（如附加值服务或产品）？
# 6. 成本结构：核心的成本支出是什么？如何通过效率提升或其他方式控制成本？
# 7. 供应链管理：新式茶饮业的供应链是怎样的？优质的原材料是如何采购的？物流和储存不同茶饮配料的方式又是如何？
# 8. 创新元素：新式茶饮业有哪些创新的影响力或有潜力的商业模式？这些创新如何推动企业的持续增长？
# 9. 竞争环境：在新式茶饮行业中，主要的竞争对手是哪些？他们的成功因素或失败因素是什么？有无可能出现新的竞争者或潜在威胁？
# 10. 商业可持续性：新式茶饮业对环境和社会的影响如何进行管理和减少？有无其他可持续发展的商业做法或潜在机会？
# 这些问题和角度可以帮助您更全面地思考和完善新式茶饮商业BP的商业模式部分。通过分析和回答这些问题，您可以深入了解茶饮行业并从中获取灵感和创意。记得和商业案例数据库中的相关案例相比较和分析，以寻找最适合您的商业模式。


def find_file_ext(_dir, _ext):
    _files = []
    for root, dirs, files in os.walk(_dir):
        for file in files:
            if file.endswith(_ext):
                full_path = os.path.join(root, file)
                _files.append(full_path)
    return _files


def extract_questions(text):
    pattern = r'\d+\.\s*[^：]+[：]\s*([^。？]*[。？])'
    extracted_texts = re.findall(pattern, text)
    split_lists = [re.findall(r'[^。？]*[。？]', text) for text in extracted_texts]
    flat_list = [item.strip() for sublist in split_lists for item in sublist]
    return flat_list


def qa_faiss_openai(_query, _db_name):
    start_time = time.time()
    _vdb = get_faiss_OpenAI(_db_name)
    _ans, _step, _token_cost = qa_vdb_multi_query(_query, _vdb, 'stuff')
    end_time = time.time()
    execution_time = end_time - start_time
    _time = f"Time: {execution_time:.1f} seconds"
    logger_toolset.info(f"{_time}")
    return [_ans, _step, _token_cost, _time]


def clean_txt(_txt):
    _txt = re.sub(r"\n+", "\n", _txt)
    _txt = re.sub(r"\t+", "\t", _txt)
    _txt = re.sub(r' +', ' ', _txt)
    _txt = re.sub(r'^\s+', '', _txt, flags=re.MULTILINE)
    return _txt


def qlist_to_ans(_dir, _db_name):
    _files = find_file_ext(_dir, '.qlist')
    for i in _files:
        _topic = os.path.basename(i).split('.')[0]
        logger_toolset.info(f"{_topic}")
        _out_fp = os.path.join(_dir, f"{_topic}.ans")
        _out = []
        _out.append(f"# {_topic}\n")
        with open(i, 'r') as rf:
            _qlist = rf.read()
        _questions = extract_questions(_qlist)
        for j in _questions:
            _q = f"{j}" + "如果未找到相关答案，仅输出'无'。"
            _ans, _step, _token_cost, _time = qa_faiss_openai(_q, _db_name)
            logger_toolset.info(f"'{_ans}'")
            # logger_toolset.info(f"{_step}")
            logger_toolset.info(f"{_token_cost}")
            logger_toolset.info(f"{_time}")
            if _ans != '无':
                if _ans.endswith('无'):
                    _ans = _ans.replace("无", "")
                _ans = clean_txt(_ans)
                _out.append(f"## {j}\n{_ans}\n")
        with open(_out_fp, 'w') as wf:
            wf.write('\n'.join(_out))


def bp_chain(_sys, _text):
    _re, _token_cost = "", ""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _sys),
            ("human", "{text}"),
        ]
    )
    model = ChatOpenAI(model="gpt-4", temperature=0)
    runnable = (
        {"text": RunnablePassthrough()}
        | prompt 
        | model 
        | StrOutputParser()
    )
    with get_openai_callback() as cb:
        _re = runnable.invoke(_text)
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
    return _re, _token_cost


SLIDE_TEMPLATE = """# MISSION
You are a slide deck builder. You will be given a topic and will be expected to generate slide deck text with a very specific format. 

# INPUT
The user will give you input of various kinds, usually a topic or request. This will be highly varied, but your output must be super consistent.

# OUTPUT FORMAT

1. Slide Title (Two or Three Words Max)
2. Concept Description of Definition (2 or 3 complete sentences with word economy)
3. Exactly five points, characteristics, or details in "labeled list" bullet point format

# EXAMPLE OUTPUT

Speed Chess

Speed chess is a variant of chess where players have to make quick decisions. The strategy is not about making perfect moves, but about making decisions that are fractionally better than your opponent's. Speed is more important than perfection.

- Quick Decisions: The need to make moves within a short time frame.
- Fractionally Better Moves: The goal is not perfection, but outperforming the opponent.
- Speed Over Perfection: Fast, good-enough decisions are more valuable than slow, perfect ones.
- Time Management: Effective use of the limited time is crucial.
- Adaptability: Ability to quickly adjust strategy based on the opponent's moves."""


SLIDE_TEMPLATE_ZH = """＃ 使命
您是幻灯片制作者。 您将获得一个主题，并需要生成具有非常特定格式的幻灯片文本。

＃ 输入
用户将为您提供各种输入，通常是主题或请求。 这将会有很大的不同，但你的输出必须非常一致。

＃ 输出格式

1. 幻灯片标题（最多五六个字）
2. 定义的概念描述（2或3个带有经济词的完整句子）
3. “标记列表”项目符号格式中恰好有五个点、特征或细节

# 输出示例

速度棋

速度棋是国际象棋的一种变体，玩家必须快速做出决定。 该策略不是要做出完美的动作，而是要做出比对手好一点的决策。 速度比完美更重要。

- 快速决策：需要在短时间内采取行动。
- 分数更好的动作：目标不是完美，而是超越对手。
- 速度胜于完美：快速、足够好的决策比缓慢、完美的决策更有价值。
- 时间管理：有效利用有限的时间至关重要。
- 适应性：能够根据对手的动作快速调整策略。"""


def ans_to_bp(_dir, _category):
    _files = find_file_ext(_dir, '.ans')
    for i in _files:
        _topic = os.path.basename(i).split('.')[0]
        _topic_goal = _bp_10_zh[_topic]
        logger_toolset.info(f"{_topic}")
        _out_fp = os.path.join(_dir, f"{_topic}.bp")
        with open(i, 'r') as rf:
            _ans = rf.read()
        _text = f"你是一个经验丰富的'{_category}'赛道的创业者。请针对红杉资本类型的机构投资人，结合以下内容撰写商业计划书的'{_topic}'({_topic_goal})部分:\n\n{_ans}"
        _re, _token_cost = bp_chain(SLIDE_TEMPLATE_ZH, _text)
        logger_toolset.info(f"'{_re}'")
        logger_toolset.info(f"{_token_cost}")
        with open(_out_fp, 'w') as wf:
            wf.write(_re)


def bp_to_md(_dir, _category):
    _out_fp = os.path.join(_dir, f"{_category}.md")
    _out = []
    _files = find_file_ext(_dir, '.bp')
    _pages = [
        "公司宗旨",
        "市场痛点",
        "解决方案",
        "时机",
        "市场空间",
        "竞争态势",
        "产品",
        "商业模式",
        "团队",
        "财务预测",
        "愿景",
    ]
    for i in _pages:
        for j in _files:
            if i in j:
                with open(j, 'r') as rf:
                    _text = rf.read()
                _text = _text.split("\n\n")[1:]
                _text = "\n\n".join(_text)
                _out.append(f"# {i}\n\n{_text}")
    with open(_out_fp, 'w') as wf:
        wf.write('\n\n'.join(_out))


bp_schema_urls = [
    "https://mp.weixin.qq.com/s?__biz=MzAwODE5NDg3NQ==&mid=2651241894&idx=1&sn=cef55d78e8358fcb112161909deedf9b&chksm=808083f2b7f70ae44fb787bd7a3f989d0bde44f8dca2c9084dbebc56b81dceaffd3d83a40e13&mpshare=1&scene=1&srcid=0712rUvBd2qwR1f8J4NiuRVn&sharer_sharetime=1689138875790&sharer_shareid=e232e219e77fde9c69a9fd7891294beb&exportkey=n_ChQIAhIQushYMwnOzvA2rZJC9eMObBLfAQIE97dBBAEAAAAAAChQIh67qFgAAAAOpnltbLcz9gKNyK89dVj0OVwZPhDTnMcgj8QsTg3Awd7xv1V4QZqD0C%2BkRFtFYa4QjhWuMAawvFLparvqWKEbla2GyZOByt8f6UJsMuZnul1mwzfFSN3YvsZh%2FKlH7YB5JXxQOCrg9iQGDylEs8x83lgwVYCe7MI8fUpCRk%2FUKj6LxaMCTMx0VTtp6cFdCOG1ch7eoZXD3dwfhkGN5j0nmV6YEu4zVcRMNdkVa%2BSETpi9SKDNsZOgVsN4cbNaLrj0hoYRcO%2BBNz8%3D&acctmode=0&pass_ticket=QlGoQ3abrBrScGv3K%2BtjIY49pxaeULaYhlaoWtKwSF5kgB%2FXO10zgrGdnHnOZvNC&wx_header=0#rd",
    "https://mp.weixin.qq.com/s?__biz=MzkzOTE3Mjc0Mw==&mid=2247485653&idx=2&sn=2ba3ae266d59928d623790676cb330f0&chksm=c2f5be3df582372b3542d778a132e9c6a8e4a16dff72f29e4cab7d56481d763bfac81ff72ad7&mpshare=1&scene=1&srcid=10249EmVRr2OIkTCcCwymiGg&sharer_shareinfo=ff48ad6cce39dbd48f12ca020f2ca665&sharer_shareinfo_first=ff48ad6cce39dbd48f12ca020f2ca665&exportkey=n_ChQIAhIQWm18TtzqQgmxzKUXfFqhohLfAQIE97dBBAEAAAAAAHSDLkEz4TwAAAAOpnltbLcz9gKNyK89dVj0JQQqNWWeXN2%2FDEspQcDqe8yV7ZjdvW3eCyKHrbFwqg%2F1AXbZVlVxTnW5MIQrt%2BGXpcSOMZee65RV85WDt0oPe%2BUfqlpYPcEoF1sjLNYR%2F9OGesIimFnmEX6G4wB0UNJ0NXoLu8plFCgGRQzaFlUjSYgpOKkHVdcaFQLw%2BWZZmqO1hr1DDv%2BmQ0TZg0KJbZ8JXAg7sbahTSmiHIGaBVszNyxk4CNoFGpnRl2Whinwzo5jGT7r%2BLz1JfQ%3D&acctmode=0&pass_ticket=UCWH4gnvTTUTRtR9fTTdP0SCRnVWTPM%2B50MhWwlo4SGvEFOnHo8GHxVfDmOgzhYk&wx_header=0#rd",
    "https://mp.weixin.qq.com/s?__biz=MzIzMDI2ODQyNQ==&mid=2247491537&idx=1&sn=dd3024e6f45499e5e449af0150a68980&chksm=e8b754b7dfc0dda1573949a4ca442accc6f947fadeaa63ec83914a285567f7650dfbce8e7014&mpshare=1&scene=1&srcid=1024k726iTRWDL76R19WHyKU&sharer_shareinfo=46ae74af33674252763fb57bf3393f37&sharer_shareinfo_first=46ae74af33674252763fb57bf3393f37&exportkey=n_ChQIAhIQBbLLOErYAqhTTbp%2Bv8oOuxLvAQIE97dBBAEAAAAAANBcLZjh2T8AAAAOpnltbLcz9gKNyK89dVj0LKQB1Q07ORc7PQ3inRPsDr9HXw%2B6LPd9fPeHWNWca7MDl3pCUOKmPO80OtQTq%2BEZpaj5vDG5m%2Fs33PDOgWU4Uw%2Bth0YBJsWz1cMY8iZu%2BzLVDT0oXM8EE455pms2o%2Bltk27tbvSqyOTtHeeMAStbYibdiWydVwjxXxF8zebNHPJuZbb0h9qQCrX7mYftCfEuj1iTIwqGUpR%2B%2BJarLR8%2Bf41Qp51A8FxZnMFc3E%2FKTwKImcakA4mGL3P1p2BwTrqaVNufOHzfAHdI&acctmode=0&pass_ticket=Rcd%2FZr3rGEN5PVPhxCms7S0GGBmjZXfxUvmbzB22Lakg%2BFt7diGeqKzn6KXtl%2Bme&wx_header=0#rd",
]


def create_bp_from_urls(pj_urls: list[str], _category: str, bp_schema_urls=bp_schema_urls):
    _dir = str(WORKSPACE_TOOLSET / _category)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    vdb_pj = str(WORKSPACE_TOOLSET / f"{_category}_pj_openai")
    vdb_bp = str(WORKSPACE_TOOLSET / f"{_category}_schema_openai")
    urls_to_faiss_OpenAI(pj_urls, vdb_pj, str(WORKSPACE_TOOLSET / f"{_category}_pj_clean_txt"))
    urls_to_faiss_OpenAI(bp_schema_urls, vdb_bp, str(WORKSPACE_TOOLSET / f"{_category}_schema_clean_txt"))
    for _topic in _bp_10_zh.keys():
        _re, _token_cost = get_inspiration(_category, _topic)
        with open(os.path.join(_dir, f"{_topic}.qlist"), "w") as f:
            f.write(f"{_re}\n\n{_token_cost}")
    qlist_to_ans(_dir, vdb_pj)
    ans_to_bp(_dir, _category)
    bp_to_md(_dir, _category)

