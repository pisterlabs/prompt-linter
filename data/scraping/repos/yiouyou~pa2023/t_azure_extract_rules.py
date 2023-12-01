def writeF(_dir, _fn, _txt):
    import os
    wfn = os.path.join(_dir, _fn)
    with open(wfn, 'w', encoding='utf-8') as wf:
        wf.write(_txt)


def readF(_dir, _fn):
    import os
    rfn = os.path.join(_dir, _fn)
    with open(rfn, 'r', encoding='utf-8') as rf:
        return rf.read()


def _chatmodel(_ans_str, _dir, _service):
    _info = _ans_str
    _out_rule = '_rule'
    _out_step = '_rule_step'
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.callbacks import get_openai_callback
    from langchain.prompts.prompt import PromptTemplate
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.chat_models import JinaChat
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    sys_template = (
        " You are a cost optimization expert, providing cost optimization suggestions for Azure cloud service customers. In order to achieve this goal, it is necessary to first construct a list of cost optimization rules, listing what can and cannot be done in various situations; then write python code according to the cost optimization rules, which is related to inputting the usage status of customer cloud services When using data, all feasible optimization measures can be directly calculated and recommended with priority of cost and safety."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    _sB = """
Giving the below FAQ document on {service} :
--------------------
{info}
--------------------

Giving some exmpale rules:
--------------------
1. Ultra disks can only be created as empty data disks and cannot be used as OS disks.
2. Azure Backup and Azure Site Recovery do not support Ultra disks.
3. Ultra disks support a 4k physical sector size by default, but a 512E sector size is also available.
4. The only infrastructure redundancy option currently available for Ultra disks is availability zones. VMs using any other redundancy options cannot attach an Ultra disk.
5. Ultra disks cannot be shared across availability zones.
6. Premium SSD v2 currently only supports locally redundant storage (LRS) configurations. It does not support zone-redundant storage (ZRS) or other redundancy options.
7. Standard SSDs only support 512E sector size.
8. Standard HDDs only support locally redundant storage (LRS) and have a sector size of 512E.
--------------------

Based on those information, first extract all information related to cost drivers, pricing options, performance metrics, related key concepts (if any, maybe one of 'service tier', 'compute tier', 'purchase model', 'hardware type', 'storage', 'backup storage', 'long-term retention', 'locally redundant storage', 'redundancy', etc.) and the subcategories of related key concepts.
"""
    _s = """
Then, extract all possible rules that might lead, affect or limit to cost savings.
"""
# Then, provide detailed information on how cost derivers work, and how these can contribute to cost savings.
# Then, provide detailed information on how pricing options work, and how these can contribute to cost savings.
# Then, provide detailed information on how performance metrics work, and how these can contribute to cost savings.
# Then, provide detailed information on cost-saving features of subcategories of those key concepts, and how these can contribute to cost savings.
# Then, provide detailed information on unique features of subcategories of those key concepts, and how these can contribute to cost savings.
# Then, provide detailed information on restrictions of cost savings.
# Furthermore, extract all detailed cases where changing available services or plans or tiers or models or options, that might lead to cost savings.
    _sE = """
Compile all this information into a comprehensive cost optimization rule book for writing Python programs for Azure users.
"""
    human_template = _sB + _s
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    rule_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    import os, re
    with get_openai_callback() as cb:
        # llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0)
        chain = LLMChain(llm=llm, prompt=rule_prompt)
        _re = chain.run(info=_info, service=_service)
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        _rule = _re.strip().split("\n")
        _step4 = f"{_token_cost}\n\n" + "="*20+" prompt "+"="*20+"\n" + rule_prompt.format(info=_info, service=_service) + "="*20+" prompt "+"="*20+"\n" + f"extracted rules:\n\n" + "\n".join(_rule)
    _rule_str = "\n".join(_rule)
    print(_rule_str)
    writeF(_dir, _out_rule, _rule_str)
    writeF(_dir, _out_step, _step4)


def extract_rules(_ans_f, _dir, _service):
    with open(_ans_f, 'r', encoding='utf-8') as rf:
        _ans_str = rf.read()
    _chatmodel(_ans_str, _dir, _service)



if __name__ == "__main__":

    import os, sys, json

    _ans_f = os.path.join(sys.argv[1], '_ans')
    extract_rules(_ans_f, sys.argv[1], sys.argv[2])
    # python t_azure_get_qlist_from_json.py tmp_azure_managed_disk_1690514432 "azure managed disk"

