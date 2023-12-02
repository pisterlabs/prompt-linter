import os

from dotenv import load_dotenv
from langchain.chains import ConversationChain, LLMChain, LLMRouterChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

load_dotenv()


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BUG_STEP1_PROMPT_TEMPLATE = os.path.join(
    CUR_DIR, "prompt_templates", "bug_say_sorry.txt"
)
BUG_STEP2_PROMPT_TEMPLATE = os.path.join(
    CUR_DIR, "prompt_templates", "bug_request_context.txt"
)
ENHANCE_STEP1_PROMPT_TEMPLATE = os.path.join(
    CUR_DIR, "prompt_templates", "enhancement_say_thanks.txt"
)


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )


llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

bug_step1_chain = create_chain(
    llm=llm,
    template_path=BUG_STEP1_PROMPT_TEMPLATE,
    output_key="text",
)
enhance_step1_chain = create_chain(
    llm=llm,
    template_path=ENHANCE_STEP1_PROMPT_TEMPLATE,
    output_key="text",
)


destinations = [
    "bug: Related to a bug, vulnerability, unexpected error with an existing feature",
    "documentation: Changes to documentation and examples, like .md, .rst, .ipynb files. Changes to the docs/ folder",
    "enhancement: A large net-new component, integration, or chain. Use sparingly. The largest features",
    "improvement: Medium size change to existing code to handle new use-cases",
    "nit: Small modifications/deletions, fixes, deps or improvements to existing code or docs",
    "question: A specific question about the codebase, product, project, or how to use a feature",
    "refactor: A large refactor of a feature(s) or restructuring of many files",
]
destinations = "\n".join(destinations)
router_prompt_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations)
router_prompt = PromptTemplate.from_template(
    template=router_prompt_template, output_parser=RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt, verbose=True)

multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains={
        "bug": bug_step1_chain,
        "enhancement": enhance_step1_chain,
    },
    default_chain=ConversationChain(llm=llm, output_key="text"),
)
