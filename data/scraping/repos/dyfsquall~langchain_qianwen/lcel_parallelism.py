from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel

from langchain_qianwen import Qwen_v1

if __name__ == "__main__":
    jock_template = "给我讲个有关 {topic} 的笑话"
    story_template = "写一个简短的关于 {story} 的故事"

    prompt1 = PromptTemplate.from_template(jock_template)
    prompt2 = PromptTemplate.from_template(story_template)

    llm = Qwen_v1(
        model_name="qwen-turbo",
        temperature=0.9,
    )

    chain1 = prompt1 | llm
    chain2 = prompt2 | llm

    combined = RunnableParallel(joke=chain1, story=chain2)
    combined.invoke({"topic": "UI小红帽", "story": "产品大灰狼"})
