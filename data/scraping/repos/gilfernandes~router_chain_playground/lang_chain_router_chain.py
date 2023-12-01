from langchain.chains.router import MultiPromptChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

from prompt_toolkit import HTML, prompt
import langchain.callbacks

langchain.callbacks.StdOutCallbackHandler

from FileCallbackHandler import FileCallbackHandler

from pathlib import Path

file_ballback_handler = FileCallbackHandler(Path('router_chain.txt'), print_prompts=True)

class Config(): 
    model = 'gpt-3.5-turbo-0613'
    llm = ChatOpenAI(model=model, temperature=0, callbacks=[file_ballback_handler])

cfg = Config()

class PromptFactory():
    developer_template = """You are a very smart Python programmer. \
    You provide answers for algorithmic and computer problems in Python. \
    You explain the code in a detailed manner. \

    Here is a question:
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



    prompt_infos = [
        {
            'name': 'python programmer',
            'description': 'Good for questions about coding and algorithms',
            'prompt_template': developer_template
        },
        {
            'name': 'poet',
            'description': 'Good for generating poems for creative questions',
            'prompt_template': poet_template
        },
        {
            'name': 'wikipedia expert',
            'description': 'Good for answering questions about general knowledge',
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
        }
    ]



def generate_destination_chains():
    """
    Creates a list of LLM chains with different prompt templates.
    """
    prompt_factory = PromptFactory()
    destination_chains = {}
    for p_info in prompt_factory.prompt_infos:
        name = p_info['name']
        prompt_template = p_info['prompt_template']
        chain = LLMChain(
            llm=cfg.llm, 
            prompt=PromptTemplate(template=prompt_template, input_variables=['input']))
        destination_chains[name] = chain
    default_chain = ConversationChain(llm=cfg.llm, output_key="text")
    return prompt_factory.prompt_infos, destination_chains, default_chain


def generate_router_chain(prompt_infos, destination_chains, default_chain):
    """
    Generats the router chains from the prompt infos.
    :param prompt_infos The prompt informations generated above.
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
    return MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
        callbacks=[file_ballback_handler]
    )
    


if __name__ == "__main__":
    # Put here your API key or define it in your environment
    # os.environ["OPENAI_API_KEY"] = '<key>'

    prompt_infos, destination_chains, default_chain = generate_destination_chains()
    chain = generate_router_chain(prompt_infos, destination_chains, default_chain)
    while True:
        question = prompt(
            HTML("<b>Type <u>Your question</u></b>  ('q' to exit, 's' to save to html file): ")
        )
        if question == 'q':
            break
        if question == 's':
            file_ballback_handler.create_html()
            continue
        result = chain.run(question)
        print(result)
        print()

    