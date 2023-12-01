import pprint as pp


key = "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"
N_batch = 5


def call_openai(chain, _content, _empty):
    from langchain.callbacks import get_openai_callback
    _re = ""
    _tokens = 0
    _cost = 0
    _log = ""
    with get_openai_callback() as cb:
        _re = chain.run(_content=_content, _empty=_empty)
        _tokens = cb.total_tokens
        _cost = cb.total_cost
        _log += f"\nTokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})\n"
        _log += f"Cost: ${cb.total_cost}\n\n"
    # print(_re)
    return [_re, _tokens, _cost, _log]


def competitor_openai(key, txt_lines, N_batch):
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
Ignore previous instructions. As a business competitor analyst, your task is to identify and extract the specific names or brands of competitors (well-known organizations or companies) from each customer comment.

The customer comments that require marketing strategy analysis are as follows:
{_content}

The output form of the final result is an array [{_empty}, {_empty},...], each element in this array is the analysis result corresponding to each comment, and the order of the elements is given order of comments. The analysis result of each comment is a competing product analysis result in JSON format, with the 'competitors' the main key and the identified competitors as the values. If no information is available for 'competitors', set its value as an empty string.
Please output the analysis results in English lowercase and only the array:
"""
    prompt = PromptTemplate(
        input_variables=["_content", "_empty"],
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    ##### split comment to sentences
    _sentences = []
    for i in txt_lines:
        i_li = i.strip()
        if i_li:
            for j in i_li.split(". "):
                jj = ""
                if j[-1] == '.':
                    jj = j
                else:
                    jj = j+"."
                _sentences.append(jj)
    ##### call OpenAI API with _content
    all_re = ""
    for i in range(0, len(_sentences)):
        if i % N_batch == 0:
            batch = _sentences[i:i+N_batch]
            # print(batch)
            _content = ""
            n = int(i / N_batch)
            for j in range(0, len(batch)):
                _content = _content + f"{n*N_batch +j +1}) {batch[j]}\n"
            _log += _content
            # print(prompt.format(_content=_content, _empty=r"{}"))
            [b_re, b_tokens, b_cost, b_log] = call_openai(chain, _content, r"{}")
            _log += b_log
            _total_cost += b_cost
            all_re += b_re + "\n"
            # print(b_re)
    _total_cost_str = format(_total_cost, ".5f")
    _competitor_str = all_re.strip()
    return [_log, _competitor_str, _total_cost_str]


def competitor_llm(_txt):
    global key
    _log = ""
    _competitor_str = ""
    _total_cost = 0
    txt_lines = _txt.split("\n")
    [_log, _competitor_str, _total_cost_str] = competitor_openai(key, txt_lines, N_batch)
    # print(_log)
    # print(_competitor_str)
    import ast
    _competitor = []
    _competitor_str_array = _competitor_str.split("\n")
    for i in _competitor_str_array:
        _i = ast.literal_eval(i)
        _competitor.extend(_i)    
    # print(type(_competitor), _competitor)
    return [_competitor, _total_cost_str]



if __name__ == "__main__":

    _txt = "Ved ikke om de har noget organisk affald... på deres hovedkontor har de et køkken, men det er en ekstern operatør der driver det... det er Michael Kjær fra driften, et fælles køkken med andre virksomheder.. Ring til ham om det. NCC bestemmer desuden selv om de skal have vores projekt med i loopet på dgnb point i byggeriet... i deres koncept udvikling...; De er ved at definere det og vi kan vende retur til Martin i Januar, hvor han ved hvem vi skal have møde med om det."
    [_re, _cost] = competitor_llm(_txt)
    print(type(_re), _re)
    print(type(_cost), _cost)
