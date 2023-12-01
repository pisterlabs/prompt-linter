from llm.llm import llm
from langchain.chains import LLMChain
from prompt.keyword_recommendation import KEYWORD_PROMPT
from langchain.prompts import PromptTemplate


class KeyWordRecommendationServices:

    @classmethod
    def recommend_keywords(cls, topic):
        prompt1 = PromptTemplate.from_template(KEYWORD_PROMPT)
        chain = LLMChain(prompt=prompt1, llm=llm, verbose=True)
        keywords = chain.run(topic)
        keywords = keywords.replace(".", "").replace("。", "")
        # chain2 = LLMChain(prompt=prompt, llm=llm, verbose=True)
        return
