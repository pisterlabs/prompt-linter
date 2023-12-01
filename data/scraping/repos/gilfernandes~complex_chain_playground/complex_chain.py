from langchain.chains.router import MultiRouteChain, RouterChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import SimpleSequentialChain, TransformChain

from prompt_toolkit import HTML, prompt
import langchain.callbacks

from replace_function import create_transform_func

langchain.callbacks.StdOutCallbackHandler

from FileCallbackHandler import FileCallbackHandler

from pathlib import Path
from typing import Mapping, List, Union

file_ballback_handler = FileCallbackHandler(Path('router_chain.txt'), print_prompts=True)

class Config(): 
    model = 'gpt-3.5-turbo-0613'
    llm = ChatOpenAI(model=model, temperature=0)

cfg = Config()

class PromptFactory():
    developer_template = """You are a very smart Python programmer. \
    You provide answers for algorithmic and computer problems in Python. \
    You explain the code in a detailed manner. \

    Here is a question:
    {input}"""

    python_test_developer_template = """You are a very smart Python programmer who writes unit tests using pytest. \
    You provide test functions written in pytest with asserts. \
    You explain the code in a detailed manner. \

    Here is a input on which you create a test:
    {input}"""

    kotlin_developer_template = """You are a very smart Kotlin programmer. \
    You provide answers for algorithmic and computer science problems in Kotlin. \
    You explain the code in a detailed manner. \

    Here is a question:
    {input}"""

    kotlin_test_developer_template = """You are a very smart Kotlin programmer who writes unit tests using JUnit 5. \
    You provide test functions written in JUnit 5 with JUnit asserts. \
    You explain the code in a detailed manner. \

    Here is a input on which you create a test:
    {input}"""

    poet_template = """You are a poet who replies to creative requests with poems in English. \
    You provide answers which are poems in the style of Lord Byron or Shakespeare. \

    Here is a question:
    {input}"""

    wiki_template = """You are a Wikipedia expert. \
    You answer common knowledge questions based on Wikipedia knowledge. \
    Your explanations are detailed and in plain English.

    Here is a question:
    {input}"""

    image_creator_template = """You create a creator of images. \
    You provide graphic representations of answers using SVG images.

    Here is a question:
    {input}"""

    legal_expert_template = """You are a UK or US legal expert. \
    You explain questions related to the UK or US legal systems in an accessible language \
    with a good number of examples.

    Here is a question:
    {input}"""

    word_filler = """Your job is to fill the words in a sentence in which words seems to be missing.
    
    Here is the input:
    {input}"""

    python_programmer = 'python programmer'
    kotlin_programmer = 'kotlin programmer'

    programmer_test_dict = {
        python_programmer: python_test_developer_template,
        kotlin_programmer: kotlin_test_developer_template
    }

    word_filler_name = 'word filler'

    prompt_infos = [
        {
            'name': python_programmer,
            'description': 'Good for questions about coding and algorithms in Python',
            'prompt_template': developer_template
        },
        {
            'name': 'python tester',
            'description': 'Good for for generating Python tests from existing Python code',
            'prompt_template': python_test_developer_template
        },
        {
            'name': kotlin_programmer,
            'description': 'Good for questions about coding and algorithms in Kotlin',
            'prompt_template': kotlin_developer_template
        },
        {
            'name': 'kotlin tester',
            'description': 'Good for for generating Kotlin tests from existing Kotlin code',
            'prompt_template': kotlin_test_developer_template
        },
        {
            'name': 'poet',
            'description': 'Good for generating poems for creatinve questions',
            'prompt_template': poet_template
        },
        {
            'name': 'wikipedia expert',
            'description': 'Good for answering questions about general knwoledge',
            'prompt_template': wiki_template
        },
        {
            'name': 'graphical artist',
            'description': 'Good for answering questions which require an image output',
            'prompt_template': image_creator_template
        },
        {
            'name': 'legal expert',
            'description': 'Good for answering questions which are related to UK or US law',
            'prompt_template': legal_expert_template
        },
        {
            'name': word_filler_name,
            'description': 'Good at filling words in sentences with missing words',
            'prompt_template': word_filler
        }
    ]


class MyMultiPromptChain(MultiRouteChain):
    """A multi-route chain that uses an LLM router chain to choose amongst prompts."""

    router_chain: RouterChain
    """Chain for deciding a destination chain and the input to it."""
    destination_chains: Mapping[str, Union[LLMChain, SimpleSequentialChain]]
    """Map of name to candidate chains that inputs can be routed to."""
    default_chain: LLMChain
    """Default chain to use when router doesn't map input to one of the destinations."""

    @property
    def output_keys(self) -> List[str]:
        return ["text"]


def generate_destination_chains():
    """
    Creates a list of LLM chains with different prompt templates.
    Note that some of the chains are sequential chains which are supposed to generate unit tests.
    """
    prompt_factory = PromptFactory()
    destination_chains = {}
    for p_info in prompt_factory.prompt_infos:
        name = p_info['name']
        prompt_template = p_info['prompt_template']
        
        chain = LLMChain(
            llm=cfg.llm, 
            prompt=PromptTemplate(template=prompt_template, input_variables=['input']),
            output_key='text',
            callbacks=[file_ballback_handler]
        )
        if name not in prompt_factory.programmer_test_dict.keys() and name != prompt_factory.word_filler_name:
            destination_chains[name] = chain
        elif name == prompt_factory.word_filler_name:
            transform_chain = TransformChain(
                input_variables=["input"], output_variables=["input"], transform=create_transform_func(3), callbacks=[file_ballback_handler]
            )
            destination_chains[name] = SimpleSequentialChain(
                chains=[transform_chain, chain], verbose=True, output_key='text', callbacks=[file_ballback_handler]
            )
        else:
            # Normal chain is used to generate code
            # Additional chain to generate unit tests
            template = prompt_factory.programmer_test_dict[name]
            prompt_template = PromptTemplate(input_variables=["input"], template=template)
            test_chain = LLMChain(llm=cfg.llm, prompt=prompt_template, output_key='text', callbacks=[file_ballback_handler])
            destination_chains[name] = SimpleSequentialChain(
                chains=[chain, test_chain], verbose=True, output_key='text', callbacks=[file_ballback_handler]
            )


    default_chain = ConversationChain(llm=cfg.llm, output_key="text")
    return prompt_factory.prompt_infos, destination_chains, default_chain


def generate_router_chain(prompt_infos, destination_chains, default_chain):
    """
    Generats the router chains from the prompt infos.
    :param prompt_infos The prompt informations generated above.
    :param destination_chains The LLM chains with different prompt templates
    :param default_chain A default chain
    """
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = '\n'.join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=['input'],
        output_parser=RouterOutputParser()
    )
    router_chain = LLMRouterChain.from_llm(cfg.llm, router_prompt)
    multi_route_chain = MyMultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True, 
        callbacks=[file_ballback_handler]
    )
    return multi_route_chain
    


if __name__ == "__main__":
    # Put here your API key or define it in your environment
    # os.environ["OPENAI_API_KEY"] = '<key>'

    prompt_infos, destination_chains, default_chain = generate_destination_chains()
    chain = generate_router_chain(prompt_infos, destination_chains, default_chain)
    with open('conversation.log', 'w') as f:
        while True:
            question = prompt(
                HTML("<b>Type <u>Your question</u></b>  ('q' to exit, 's' to save to html file): ")
            )
            if question == 'q':
                break
            if question in ['s', 'w'] :
                file_ballback_handler.create_html()
                continue
            result = chain.run(question)
            f.write(f"Q: {question}\n\n")
            f.write(f"A: {result}")
            f.write('\n\n ====================================================================== \n\n')
            print(result)
            print()

    