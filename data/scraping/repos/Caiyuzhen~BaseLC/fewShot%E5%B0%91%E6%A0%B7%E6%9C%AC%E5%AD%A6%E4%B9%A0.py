from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate


"""
	少样本学习
		少样本学习不会改变模型的结构，而是改变模型的参数，使得模型能够适应新的任务
 		少样本学习的目标是在少量样本的情况下，让模型具有较强的泛化能力
"""

EXAMPLES = [
	{
		"question": "你好吗?",
		"answer": "当然!我很好!"
	},
 	{
 		"question": "今天天气怎么样?",
		"answer": "当然!天气很不错!"
	},
  	{
 		"question": "今天的食物怎么样?",
		"answer": "当然!食物很美味!"
	},
]


def get_user_input():
	input_content = input("请输入问题: ")
	return input_content


def run_llm(input_content):
    llm = OpenAI()
    
    example_prompt = PromptTemplate(
		input_variables=["question", "answer"],
		template="Question: {question}\n{answer}" # 🔥相当于利用上面 EXAMPLES 的数据进行格式化
	)
    
    prompt = FewShotPromptTemplate(
		examples = EXAMPLES,
		example_prompt = example_prompt,
		suffix="Question: {input}", # 🔥 以 {input} 作为问题的输入
		input_variables=["input"]
	)

    chain = LLMChain(llm=llm, prompt=prompt)
    res = chain.run(input_content) # 🌟放入用户输入的问题
    return res


if __name__ == "__main__":
    input_content = get_user_input()
    result = run_llm(input_content)
    print(result)
