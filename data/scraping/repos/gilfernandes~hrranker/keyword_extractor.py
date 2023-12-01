from langchain.chains import create_tagging_chain_pydantic

from typing import List, Any

from hrranker.hr_model import TechnicalKeywords
from hrranker.config import cfg
from hrranker.log_init import logger


def extract_keywords(expression_list: List[str]) -> List[Any]:
    chain = create_tagging_chain_pydantic(TechnicalKeywords, cfg.llm)
    expression_pairs = []
    for expression in expression_list:
        technical_keywords: TechnicalKeywords = chain.run(expression)
        technical_keywords.keywords = [k.lower() for k in technical_keywords.keywords]
        expression_pairs.append((expression, technical_keywords.keywords))
    return expression_pairs


if __name__ == "__main__":
    expression_list = [
        "Programming in PHP",
        "Wordpress",
        "CSS",
        "Experience with HTML",
        "Figma",
    ]
    res = extract_keywords(expression_list=expression_list)
    logger.info("Result: %s", res)
    assert "php" in res[0][1]
