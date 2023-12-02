from textwrap import dedent

from fastapi import Depends
from langchain import LLMChain, OpenAI, PromptTemplate
from services.world.internal.llm import get_llm
from services.world.internal.models.world import World, get_world


def generate_world_memory(
    world: World = Depends(get_world),
    llm: OpenAI = Depends(get_llm),
) -> str:
    prompt = PromptTemplate(
        input_variables=["world_name", "description"],
        template=dedent(
            """\
        这是一个叫做 {world_name} 的幻想世界，
        
        它的简介是：
        {description}
        
        给出一件在这个世界中可能发生的事情。
        """
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(
        world_name=world.name,
        description=world.description,
    )
