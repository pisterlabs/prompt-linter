from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from asyncer import asyncify

from image_generator_bot.configuration.config import cfg
import image_generator_bot.configuration.templates as t
from image_generator_bot.configuration.log_factory import logger


async def give_prompt(description: str) -> str:
    chain = await image_chain_factory()
    prompt_generated = await chain.arun(description)
    logger.info(prompt_generated)
    return prompt_generated


def prompt_factory() -> str:
    return PromptTemplate(
        input_variables=["image_desc"],
        template=t.template_image_description,
    )


async def image_chain_factory() -> LLMChain:
    prompt = prompt_factory()
    chain = LLMChain(llm=cfg.llm, prompt=prompt, verbose=cfg.verbose_llm)
    return chain


def generate_image(prompt: str) -> str:
    return DallEAPIWrapper().run(prompt)


async def generate_advice_image(image_description) -> str:
    return await asyncify(generate_image)(image_description)


if __name__ == "__main__":
    chain = image_chain_factory()

    def create_image_bot():
        image_prompt_res = chain.run(
            {
                "image_desc": """a person holding a sheet of paper and the paper has lines 
                                      """
            }
        )
        # print(image_prompt_res)
        image_url = DallEAPIWrapper().run(image_prompt_res)
        return image_url

    print(create_image_bot())
