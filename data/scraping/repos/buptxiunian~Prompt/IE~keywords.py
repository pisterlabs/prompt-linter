# 关键词抽取
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


async def keywords_extraction(pages):
    model = "Qwen-14B-Chat-Int4"

    examples = [
        {
            "input": '''零样本信息抽取（IE）旨在从未标注的文本中构建IE系统。由于几乎不涉及人工干预，这是一个具有挑战性的任务。虽然具有挑战性，但是值得一试，零样本IE可以减少数据标注所需的时间和精力。最近对大型语言模型（LLMs，例如GPT-3、ChatGPT）的研究显示，在零样本设置下取得了令人鼓舞的性能，因此启发我们探索基于提示的方法。在这项工作中，我们探讨了是否可以通过直接提示LLMs构建强大的IE模型。具体来说，我们将零样本IE任务转化为一个多轮问答问题，并采用了两阶段框架（ChatIE）。借助ChatGPT的强大能力，我们在三个IE任务上广泛评估了我们的框架：实体关系三元组抽取、命名实体识别和事件抽取。跨两种语言的六个数据集上的实证结果显示，ChatIE取得了令人印象深刻的性能，并且甚至在某些数据集上超过了一些全样本模型（例如NYT11-HRL）。我们相信，我们的工作可能为在有限资源下构建IE模型提供启示。''',
            "output": '''{"keywords": ["零样本信息抽取","大型语言模型","提示"]}'''
        },
        {
            "input": '''情感分析（SA）一直是自然语言处理领域的一个长期研究领域。它能够提供对人类情感和意见的深入洞察，因此受到学术界和工业界的广泛关注。随着像ChatGPT这样的大型语言模型（LLMs）的出现，它们在情感分析问题上有着巨大的潜力。然而，现有的LLMs在不同情感分析任务中的利用程度仍然不清楚。本文旨在全面调查LLMs在各种情感分析任务中的能力，从传统的情感分类到基于方面的情感分析，再到主观文本的多方面分析。我们在26个数据集上评估了13个任务的性能，并将结果与在特定领域数据集上训练的小型语言模型（SLMs）进行了比较。研究发现，尽管LLMs在简单任务中表现出色，但在需要更深入理解或结构化情感信息的复杂任务中表现较差。然而，在少样本学习环境中，LLMs明显优于SLMs，表明它们在标注资源有限时的潜力。我们还指出了当前评估实践在评估LLMs的SA能力方面的局限性，并提出了一个新的基准，SENTIEVAL，用于进行更全面和现实的评估。''',
            "output": '''{"keywords": ["情感分析","自然语言处理","大型语言模型"]}'''
        },
        {
            "input": '''大型语言模型（LLMs）在语言理解和交互式决策制定任务中表现出色，但它们在推理（例如思维链提示）和行动（例如行动计划生成）方面的能力主要作为独立主题进行研究。本文探讨了LLMs在交织方式下生成推理追踪和任务特定行动的用途，允许两者之间更大的协同作用：推理追踪帮助模型引导、跟踪和更新行动计划，以及处理异常情况，而行动允许它与外部来源（如知识库或环境）进行接口交互并收集额外信息。我们提出的方法名为ReAct，在各种语言和决策任务中应用，并展示了它相对于最先进基线方法的有效性，同时提高了人类的可解释性和可信度。具体来说，在问答（HotpotQA）和事实验证（Fever）任务中，ReAct通过与简单的维基百科API进行交互，克服了思维链推理中普遍存在的幻觉和错误传播问题，并生成了更具可解释性的人类化任务解决轨迹，相较于没有推理追踪的基线方法更容易理解。此外，在两个交互式决策制定基准（ALFWorld和WebShop）上，ReAct的成功率分别比模仿学习和强化学习方法高出34%和10%，而且只需一个或两个上下文示例作为提示''',
            "output": '''{"keywords": ["大型语言模型","推理追踪","任务特定行动","交织方式","思维链提示"]}'''
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
            ("system",
             '''对输入的文本进行**关键词提取**, 要求综合考虑**已有的关键词和新的文本**，关键词**总数不超过五个**，结果并以json格式输出'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    text_splitter = RecursiveCharacterTextSplitter(
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
    res = {}
    with tqdm(total=len(pages)) as pbar:
        pbar.set_description("Processing:")
        prefix = ""
        for page in pages:
            texts = text_splitter.split_text(page.page_content)
            for text in texts:
                tmp = await chain.arun(input=prefix+text, return_only_outputs=True)
                print(tmp)
                prefix = '''下面给出目前已有的关键词列表\n'''+tmp + \
                    '''\n要求对已有的关键词和新文本进行综合考虑，总结不超过**五个**关键词,结果以**json**格式输出'''
                try:
                    json_object = json.loads(tmp)
                    res = json_object
                except Exception as e:
                    continue
            pbar.update(1)
    # 最多只返回五个关键词
    try:
        json_object = json.loads(res)
        if (len(json_object["keywords"]) > 5):
            json_object["keywords"] = json_object["keywords"][:5]
            res = json.dumps(json_object)
        return res
    except Exception as e:
        return res
