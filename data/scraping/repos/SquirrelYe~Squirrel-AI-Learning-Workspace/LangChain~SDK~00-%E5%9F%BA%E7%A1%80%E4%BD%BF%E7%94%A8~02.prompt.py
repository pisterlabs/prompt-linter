from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from langchain.callbacks import (get_openai_callback)


# 🍉 Prompt 包含如下几个概念：
# 零样本提示（Zero-Shot Prompting）
# 小样本提示（Few-Shot Prompting）
# 提示工程
# 提示模板
def generate_prompt_template():
    template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
    prompt = PromptTemplate(
        input_variables=["lastname"],
        template=template,
    )

    prompt_text = prompt.format(lastname="王")  # result: 我的邻居姓王，他生了个儿子，给他儿子起个名字
    llm = OpenAI(temperature=0.9)  # 调用OpenAI
    result = llm(prompt_text)
    print(result)

    with get_openai_callback() as cb:
        result = llm("Tell me a joke")
        print(cb)
        print(result)


def generate_prompt_template_few_shot():
    examples = [
        {
            "word": "开心",
            "antonym": "难过"
        },
        {
            "word": "高",
            "antonym": "矮"
        },
    ]

    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template="""
            单词: {word}
            反义词: {antonym}\\n
        """,
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="给出每个单词的反义词",
        suffix="单词: {input}\\n反义词:",
        input_variables=["input"],
        example_separator="\\n",
    )

    prompt_text = few_shot_prompt.format(input="粗")
    llm = OpenAI(temperature=0.9)
    print(prompt_text)
    print(llm(prompt_text))


if __name__ == '__main__':
    generate_prompt_template()
    # generate_prompt_template_few_shot()
