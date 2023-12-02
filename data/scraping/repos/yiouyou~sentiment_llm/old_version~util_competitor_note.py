import pprint as pp


key = "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"


def call_openai(chain, _content):
    from langchain.callbacks import get_openai_callback
    _re = ""
    _tokens = 0
    _cost = 0
    _log = ""
    with get_openai_callback() as cb:
        _re = chain.run(_content=_content)
        _tokens = cb.total_tokens
        _cost = cb.total_cost
        _log += f"\nTokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})\n"
        _log += f"Cost: ${cb.total_cost}\n\n"
    # print(_re)
    return [_re, _tokens, _cost, _log]


def competitor_openai(key, txt_lines):
    import os
    import re
    from langchain import OpenAI, PromptTemplate, LLMChain
    _log = ""
    _competitor_str = ""
    _total_cost = 0
    ##### set OpenAI API Key and prompt
    os.environ["OPENAI_API_KEY"] = key
    llm = OpenAI(temperature=0)
    template = """
Ignore previous instructions. As a business competitor analyst, your task is to identify and extract the names and brands of competitors (well-known businesses or brands) from customer notes.

The customer's note text that requires business competitor analysis is provided below:
{_content}

You should output the results in JSON format, with the 'competitors' the main key and the identified competitors as the values. Please output the analysis results in English lowercase:
"""
    prompt = PromptTemplate(
        input_variables=["_content"],
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    _txt_lines = "\n".join(txt_lines)
    _li = _txt_lines.strip()
    [b_re, b_tokens, b_cost, b_log] = call_openai(chain, _li)
    _log += b_log
    _competitor_str += re.sub(r"\n+", r"\n", b_re) + "\n"
    _total_cost += b_cost
    return [_log, _competitor_str, _total_cost]


def competitor_llm(_txt):
    global key
    _log = ""
    _competitor_str = ""
    _total_cost = 0
    txt_lines = _txt.split("\n")
    [_log, _competitor_str, _total_cost] = competitor_openai(key, txt_lines)
    print(_log)
    return [_competitor_str, str(_total_cost)]



if __name__ == "__main__":

    _txt = "Ved ikke om de har noget organisk affald... på deres hovedkontor har de et køkken, men det er en ekstern operatør der driver det... det er Michael Kjær fra driften, et fælles køkken med andre virksomheder.. Ring til ham om det. NCC bestemmer desuden selv om de skal have vores projekt med i loopet på dgnb point i byggeriet... i deres koncept udvikling...; De er ved at definere det og vi kan vende retur til Martin i Januar, hvor han ved hvem vi skal have møde med om det."
    print(competitor_llm(_txt))
