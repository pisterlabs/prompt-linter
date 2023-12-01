from langchain.llms import OpenAI
import warnings
warnings.filterwarnings('ignore')

def select_journals(title_):
    #model_name="gpt-4"
    model_name="gpt-3.5-turbo"
    llm = OpenAI(temperature=0, model_name=model_name)

    input = f"{title_}という論文を投稿するのに適切な英文雑誌を10個、2023年時点のインパクトファクターが高い順に並べて教えてください。各雑誌へのリンクも教えてください。"

    journals_ = llm(input)

    return journals_