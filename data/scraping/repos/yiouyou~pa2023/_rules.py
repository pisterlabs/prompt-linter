def llm_azure_rules(_text):
    _ans, _steps = "", ""
    from langchain.callbacks import get_openai_callback
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.prompts.pipeline import PipelinePromptTemplate
    from langchain.prompts.prompt import PromptTemplate
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    import os
    full_template = \
    """
    {introduction}

    {info}
    """
    full_prompt = PromptTemplate.from_template(full_template)

    introduction_template = \
    """
    You are a cost optimization expert, providing cost optimization suggestions for Azure cloud service customers. In order to achieve this goal, it is necessary to first construct a list of cost optimization rules, listing what can and cannot be done in various situations; then write python code according to the cost optimization rules, which is related to inputting the usage status of customer cloud services When using data, all feasible optimization measures can be directly calculated and recommended with priority of cost and safety.
    """
    introduction_prompt = PromptTemplate.from_template(introduction_template)

    info_template = \
    """
    Giving the following information:
    --------------------
    {info}
    --------------------
    What are necessary non-duplicative rules that you can extract to optimize the usage of Azure disks? Remember to cover as many details   as possible. Only output non-duplicative, nothing else:
    """
    info_prompt = PromptTemplate.from_template(info_template)

    input_prompts = [
        ("introduction", introduction_prompt),
        ("info", info_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    # print(pipeline_prompt.input_variables)
    with get_openai_callback() as cb:
        llm = OpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0.1)
        chain = LLMChain(llm=llm, prompt=pipeline_prompt)
        _re = chain.run(
            info=_text,
        )
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        _ans = _re.strip()
        _steps = f"{_token_cost}\n\n" + "="*20+" prompt "+"="*20+"\n" + pipeline_prompt.format(info=_text)

    return [_ans, _steps]


def chat_azure_rules(_text):
    _ans, _steps = "", ""
    from langchain.callbacks import get_openai_callback
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    sys_template = (
        "You are a cost optimization expert, providing cost optimization suggestions for Azure cloud service customers. In order to achieve this goal, it is necessary to first construct a list of cost optimization rules, listing what can and cannot be done in various situations; then write python code according to the cost optimization rules, which is related to inputting the usage status of customer cloud services When using data, all feasible optimization measures can be directly calculated and recommended with priority of cost and safety."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_template = \
    """
    Giving the following information:
    --------------------
    {info}
    --------------------
    What are necessary non-duplicative rules that you can extract to optimize the usage of Azure disks? Remember to cover as many details   as possible. Only output non-duplicative, nothing else:
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    rule_prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            human_message_prompt
        ]
    )
    import os, re
    with get_openai_callback() as cb:
        llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
        chain = LLMChain(llm=llm, prompt=rule_prompt)
        _re = chain.run(info=_text)
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.  total_cost, '.5f')}"
        _ans = _re.strip().split("\n")
        _steps = f"{_token_cost}\n\n" + "="*20+" prompt "+"="*20+"\n" + rule_prompt.format(info=_text) + "="*20+" prompt "+"="*20   +"\n" + f"{len(_ans)} rules:\n\n" + "\n".join(_ans)
    _ansH = {}
    for i in _ans:
        i_str = re.sub('^\d+\. ', '', i)
        _ansH[i_str] = 1
    _ans = "\n".join(sorted(_ansH.keys()))
    return [_ans, _steps]


if __name__ == "__main__":

    _text = """
"""
    _re1 = llm_azure_rules(_text)
    print(_re1)

    _re2 = chat_azure_rules(_text)
    print(_re2)

