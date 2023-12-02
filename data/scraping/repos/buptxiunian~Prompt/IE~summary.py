# 摘要
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


async def summary(pages):
    model = "Qwen-14B-Chat-Int4"

    examples = [
        {
            "input": '''细化实化政策。各地要结合实际，细化实化本通知明确的各项政策措施，加速释放政策红利。同步梳理前期本地出台的阶段性稳就业政策，明确优化调整意见，落实好各项常态化就业政策，推动各项政策落地见效、惠企利民，为就业大局总体稳定提供有力保障。政策实施中的重要问题和经验做法，及时报有关主管部门。
优化经办服务。各地要持续优化经办流程，减环节、减材料、减时限，编制好各项政策资金审核发放流程和办事指南。加快推进网上办理，加强大数据比对识别，推动更多政策直达快享，提升就业政策获得感和满意度。提高政策覆盖面和可及性，对符合条件的以单位形式参保的个体工商户，可参照企业同等享受就业补贴政策。规范资金管理使用，严格履行程序规定，健全风险防控机制，严肃查处骗取套取、虚报冒领等违法违规行为，保障资金安全运行。
强化宣传解读。各地要加强就业政策宣传，及时更新发布本地区就业创业政策清单，分类梳理面向高校毕业生、困难人员等不同群体和经营主体的政策举措，广泛推动稳就业政策进企业、进园区、进校园、进社区（村）。创新政策宣传方式，及时提供通俗易懂的政策解读，提高政策知晓度，稳定各方预期，营造良好社会氛围。''',
            "output": '''{"summary": "各地要细化政策，加速政策落地，推动各项政策惠及企业和个人。优化办事服务流程，提高政策的覆盖面和可及性，严格管理使用政策资金，防范违法行为。加强宣传解读，创新宣传方式，提高政策知晓度，稳定各方预期。"}'''
        },
        {
            "input": '''加强困难人员就业帮扶。合理确定并动态调整就业困难人员认定标准，及时将零就业家庭、低保家庭、脱贫户、大龄、残疾、长期失业等人员纳入援助范围。制定个性化援助方案，优先推荐低门槛、有保障的爱心岗位，提供“一对一”就业援助，对符合条件的困难毕业生发放一次性求职创业补贴。对企业招用登记失业半年以上人员，签订1年以上劳动合同的，可发放一次性吸纳就业补贴，政策实施期限截至2023年12月31日。对通过市场渠道难以实现就业的，合理统筹公益性岗位安置，确保零就业家庭至少一人就业。
（十二）保障困难群众基本生活。对符合条件的失业人员，做好失业保险金、代缴基本医疗保险费（含生育保险费）和失业农民工一次性生活补助等常规性保生活待遇发放工作。将符合条件的生活困难失业人员及家庭纳入最低生活保障、临时救助等社会救助范围。及时启动社会救助和保障标准与物价上涨挂钩联动机制，按规定向困难群众足额发放物价补贴。
四、加强组织实施''',
            "output": '''{"summary": "加强困难人员就业帮扶，提供一对一就业援助和求职创业补贴。保障困难群众基本生活，将符合条件的人员纳入最低生活保障、临时救助等社会救助范围，向困难群众发放物价补贴。"}'''
        },
        {
            "input": '''鼓励企业吸纳就业。对企业招用毕业年度或离校2年内未就业高校毕业生、登记失业的16—24岁青年，签订1年以上劳动合同的，可发放一次性吸纳就业补贴，政策实施期限截至2023年12月31日。
（七）鼓励引导基层就业。稳定“三支一扶”计划、大学生志愿服务西部计划等基层服务项目2023年招募规模。实施“大学生乡村医生”专项计划，落实医学专业高校毕业生免试申请乡村医生执业注册政策。继续做好2023年高校毕业生到城乡社区就业创业工作。对到中西部地区、艰苦边远地区、老工业基地县以下基层单位就业的高校毕业生，按规定给予学费补偿和国家助学贷款代偿、高定工资等支持，对招聘为事业单位工作人员的，可按规定提前转正定级。
（八）支持国有企业扩大招聘规模。对按照工资效益联动机制确定的工资总额难以满足扩大高校毕业生招聘需求的国有企业，经履行出资人职责机构或其他企业主管部门同意，统筹考虑企业招聘高校毕业生人数、自然减员情况和现有职工工资水平等因素，2023年可给予一次性增人增资，核增部分据实计入工资总额并作为下一年度工资总额预算基数。
（九）稳定机关事业单位岗位规模。挖掘党政机关、事业单位编制存量，统筹自然减员，加大补员力度，稳定招录、招聘高校毕业生规模，合理确定招录、招聘时间。
（十）实施2023年百万就业见习岗位募集计划。广泛动员各类企事业单位、社会组织等，募集不少于100万个青年见习岗位，对吸纳就业见习人员的给予见习补贴，用于支付见习人员基本生活费、办理人身意外伤害保险，以及对见习人员的指导管理费用。对见习期未满与见习人员签订劳动合同的，各地可给予剩余期限见习补贴，政策实施期限截至2023年12月31日。
三、强化帮扶兜牢民生底线''',
            "output": '''{'summary': "对企业招用高校毕业生等青年，提供一次性吸纳就业补贴。鼓励引导基层就业，招募规模，推动高校毕业生到城乡社区就业创业，提供相应支持。支持国有企业扩大招聘规模，给予一次性增人增资支持。实施百万就业见习岗位募集计划，提供见习补贴。"}'''
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
             '''对输入的文本进行**摘要**, 要求尽可能简洁，输出结果在200字以内，并以json格式输出，格式为：{{"summary": "摘要内容"}}'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             '''你现在已知了一篇文章的分段摘要，请将其进行整合，用流畅的语言重新进行表达.'''),
            ("human", "{input}")
        ]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=16)
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
    reduce_chain = LLMChain(
        prompt=reduce_prompt,
        # 温度调为0，可以保证输出的结果是确定的
        llm=ChatOpenAI(
            temperature=0,
            model_name=model,
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1")
        # output_parser=output_parser
    )
    map = ""
    tmp = ""
    with tqdm(total=len(pages)) as pbar:
        pbar.set_description('Processing:')
        for page in pages:
            texts = text_splitter.split_text(page.page_content)
            for text in texts:
                tmp = await chain.arun(input=text, return_only_outputs=True)
                try:
                    map += tmp
                except Exception as e:
                    continue
            pbar.update(1)
    res = reduce_chain({"input": map}, return_only_outputs=True)['text']
    print(res)
    return {"summary": res}
