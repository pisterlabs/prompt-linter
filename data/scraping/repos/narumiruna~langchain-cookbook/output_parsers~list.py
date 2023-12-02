from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from loguru import logger


def main():
    """https://langchain-langchain.vercel.app/docs/modules/chains/foundational/llm_chain"""
    load_dotenv(find_dotenv())

    llm = OpenAI(temperature=0, verbose=True)
    logger.info('llm: {}', llm)

    output_parser = CommaSeparatedListOutputParser()
    logger.info('output_parser: {}', output_parser)

    template = 'List all the colors in {object}:'
    prompt = PromptTemplate(template=template, input_variables=['object'], output_parser=output_parser)
    logger.info('prompt: {}', prompt)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    llm_result = llm_chain.predict(object='a rainbow')
    logger.info('llm_result: {}', llm_result)

    parsed_result = output_parser.parse(llm_result)
    logger.info('parsed_result: {}', parsed_result)


if __name__ == '__main__':
    main()
