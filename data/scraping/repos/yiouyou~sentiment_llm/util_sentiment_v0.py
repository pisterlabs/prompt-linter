import pprint as pp


key = "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"
N_batch = 5


def call_openai(chain, _content, _example):
    from langchain.callbacks import get_openai_callback
    _re = ""
    _tokens = 0
    _cost = 0
    _log = ""
    with get_openai_callback() as cb:
        _re = chain.run(_content=_content, _example=_example)
        _tokens = cb.total_tokens
        _cost = cb.total_cost
        _log += f"\nTokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})\n"
        _log += f"Cost: ${cb.total_cost}\n\n"
    # print(_re)
    return [_re, _tokens, _cost, _log]


def sentiment_openai(key, txt_lines, N_batch):
    import os
    import re
    from langchain import OpenAI, PromptTemplate, LLMChain
    _log = ""
    _sentences_str = ""
    _sentiments_str = ""
    _total_cost = 0
    ##### set OpenAI API Key and prompt
    os.environ["OPENAI_API_KEY"] = key
    llm = OpenAI(temperature=0)
    template = """
Ignore previous instructions. You are a sentiment analyst of customer comments. You assist the company in further operations by dividing customer comments into three categories: positive, negative and neutral. The main purpose is to judge whether customers have a positive attitude towards the products we are trying to sell to them. When analyzing comments, in addition to the general sentiment analysis principles, the following rules must be followed:
1) If the customer is likely to agree to our call back, it is considered positive
2) If the customer is willing to communicate further or is likely to purchase in the future, it is considered positive
3) If the main content of the comment involves numbers, phone numbers, dates, addresses or web addresses, it is considered neutral
4) If the main content of the comment is dominated by interjections, modal particles, nouns or adjectives with no obvious emotional meaning, it is considered neutral

Below are some examples of sentiment analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the sentiment classification of the comments:
{_example}

The customer comment texts that require sentiment analysis are as follows:
{_content}

For each comment, there is no need to output the comment itself, just output the comment index, sentiment classification and short classification reason in the format of "index) classification(reason)", and output the analysis results in English lowercase:
"""
    prompt = PromptTemplate(
        input_variables=["_content", "_example"],
        template=template,
    )
    ##### 随机取10个example
    import random
    with open("examples_sentiment.txt", "r", encoding="utf-8") as ef:
        _example = "".join(random.sample(ef.readlines(), 30))
    ##### LLMChain
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
    ##### call OpenAI API with _content and _example
    _log += "-" * 40 + "\n"
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
            # print(prompt.format(_content=_content, _example=_example))
            [b_re, b_tokens, b_cost, b_log] = call_openai(chain, _content, _example)
            _log += b_log
            _total_cost += b_cost
            all_re += b_re + "\n"
    ##### parse response, generate _log, _sentences_str and _sentiments_str
    sentences = []
    sentiments = []
    all_re = re.sub(r" *\(", " (", all_re.lower())
    all_re = re.sub(r"\n+", "\n", all_re)
    _sentiments = all_re.split("\n")
    for i in _sentiments:
        if i != "":
            sentiments.append(i)
    _log += "-" * 40 + "\n"
    _log += "\n".join(sentiments) + "\n"
    _log += "-" * 40 + "\n"
    _total_cost_str = format(_total_cost, ".5f")
    _log += f"\nTotal Cost: ${_total_cost_str}\n"
    if len(_sentences) == len(sentiments):
        for i in range(0, len(_sentences)):
            sentences.append(f"{i+1}) \"{_sentences[i]}\"")
        _sentences_str = "\n".join(sentences)
        _sentiments_str = "\n".join(sentiments)
    else:
        _log += "Error: len(sentences) != len(sentiments)" + "\n"
    return [_log, _sentences_str, _sentiments_str, _total_cost_str]


def sentiment_llm(_txt):
    global key
    import re
    _log = ""
    _sentences_str = ""
    _sentiments_str = ""
    _total_cost = 0
    txt_lines = _txt.split("\n")
    [_log, _sentences_str, _sentiments_str, _total_cost] = sentiment_openai(key, txt_lines, N_batch)
    # print(_log)
    _out = []
    if _sentences_str != "" and _sentiments_str != "":
        sentences = _sentences_str.split("\n")
        sentiments = _sentiments_str.split("\n")
        if len(sentences) == len(sentiments):
            for i in range(0, len(sentences)):
                # i_re = f"{sentences[i]}|{sentiments[i]}\n"
                _out.append(re.sub('\d+\)\s+', '', sentiments[i]))
            # print(f"return:\n{_out}")
        else:
            print("Error: len(sentences) != len(sentiments)")
    return [_out, str(_total_cost)]



if __name__ == "__main__":

    _txt = """Opfølgning på målinger i de andre butikker. Obs. på at der altid er lidt mere og obs på at det er efterårsferie, hvis det har noget at sige for omsætningen. Desuden obs på, at det er meningen, det skal opbevares i længere tid.
.med en basic og en SLIM. Den i Bones kan tages retur efter 2 måneder, den anden er købt.
Tank leveres asap Vi borer huller og trækker rør ind i bygningen Kværnen indkøbes/faktureres og leveres ca. 1/4-15
"""
    [_re, _cost] = sentiment_llm(_txt)
    print(type(_re), _re)
    print(type(_cost), _cost)
