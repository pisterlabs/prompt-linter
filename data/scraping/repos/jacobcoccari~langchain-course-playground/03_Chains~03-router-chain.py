from dotenv import load_dotenv

load_dotenv()

from langchain.chains.router import MultiPromptChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

chemistry_template = """You are a very smart chemistry professor. \
Answer the student's question concisely and clearly \ 
If you do not know the answer, say so.

Here is a question:
{input}"""

biology_template = """You are a very smart biology professor. \
Answer the student's question concisely and clearly \ 
If you do not know the answer, say so.

Here is a question:
{input}"""


prompt_infos = [
    {
        "name": "chemistry",
        "description": "Good for answering questions about chemistry",
        "prompt_template": chemistry_template,
    },
    {
        "name": "biology",
        "description": "Good for answering biology questions",
        "prompt_template": biology_template,
    },
]

model = ChatOpenAI(temperature=0)

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=model, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=model, output_key="text")

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(model, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

print(chain.run("What is black body radiation?"))
