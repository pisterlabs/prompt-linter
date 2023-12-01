

import os
import sys

from langchain.llms.base import LLM
from langchain.prompts import BasePromptTemplate, PromptTemplate
from pydantic import Field
from pydantic.dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_model import Api2dLLM


def promptFactory() -> BasePromptTemplate:
    return PromptTemplate.from_template(
        """你是一个 {role}. \n
        你在训练你所在的行业领域内的销售新人，请提供
        {num_qa} 个销售话术 Q&A 的例子. \n
        该Q&A例子的对话场景为{scenario}. \n
        请以如下格式提供例子: \n
        序号, 仅数字.
        [客户问题]
        [销售回答]
        """
    )

def modelFactory() -> LLM:
    return Api2dLLM()
@dataclass
class QAGenerator():
    """
        Generate QA pairs based on domain industry of sales man and few shots.
        For good example shots in electronic device sales, refer to [sales skills](https://zhuanlan.zhihu.com/p/357487465)
    """

    _prompt: BasePromptTemplate = Field(default_factory=promptFactory) 
    _model: LLM = Field(default_factory=modelFactory)

    @property
    def model(self) -> LLM:
        return self._model
    
    @property
    def prompt(self) -> BasePromptTemplate:
        return self._prompt

    def initQA(self, output: str):
        from langchain.chains import LLMChain
        role = "在电器行业的资深销售人员"
        scenarios = [
            "客户与销售在讨价还价", 
            "客户在询问电器产品细节",
            "客户在粗鲁地向销售抱怨"
        ]
        qa = QAGenerator()
        num_qa = 10
        chain = LLMChain(llm=qa.model, prompt=qa.prompt)
        result = [chain.run(role=role, num_qa=num_qa, scenario=s) for s in scenarios]
        with open(output, 'w', encoding='utf-8-sig') as f:
            f.writelines(result)


if __name__ == "__main__":
    qa = QAGenerator()
    qa.initQA("resources/electronic_devices_sales_qa.txt")