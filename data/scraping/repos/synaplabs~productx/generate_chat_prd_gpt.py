import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
import wandb
from wandb.integration.langchain import WandbTracer
import streamlit as st

prd_cost = 0
prd_prompt_tokens = 0
prd_completion_tokens = 0
db_cost = 0
db_prompt_tokens = 0
db_completion_tokens = 0

string_template = """\
You are a tech product manager. You have to help the user create a Product Requirement Document based on the questions the user asks you. The user will ask you specific questions about each topic they want to be included in the PRD. 

Do not repeat the same information again and again. Answers to each question should be unique and not repetitive. By this I mean do not repeat any ideas or sentences. Do not copy statements and ideas from previous sections. Any ideas or examples should only be in accordance to the particular section.

Format your responses in Markdown mode with each topic being the ##Heading, and your answer being the content. Highlight important points in **bold**. Give the PRD a suitable #Title.

For reference, let us say there are 3 people - A, B, and C belonging to different age groups, professions, and geographies. A is a 20-year-old college student from India. B is a 40-year-old working professional from the US. C is a 60-year-old retired person from the UK.
If required, for that particular section, you can use any of these people as examples to explain your point. The user does not know anything about these people.

You do not need to include these 3 people in every section. You can use them as examples only if required. You can also use other examples if you want to. You can also use yourself as an example if you want to.

Current conversation:
{history}
Human: {input}
AI: """

prompt_template = PromptTemplate(
    template=string_template,
    input_variables=["history", "input"],
)

prompts_list = [
    """Product Overview:
Define the Purpose and Scope of this product. It should include how different groups of users across ages, genders, and geographies can use this product. Include an overview of the product. Why should one use this product? Define the target audience and stakeholders in detail. Also, include the rationale behind having the particular group as the target audience. Explain the gap it is trying to fill as well - how it is different from and better than other similar products?""",
    """Product Objectives:
First, analyze whether the product objectives align with the company objectives if the company and company objectives are mentioned. Else, talk about the objectives of the product, what it will help achieve, and how it will assist customers. Think aloud. Explain your reasoning. Also, talk about why and how the business models of the product and company match. What company goals can the product help achieve - be it attracting customers, generating profits, or promoting the goodwill of the company? Also, explain how it would do this.""",
    """Launch Strategy:
Compare US vs International markets for this product. Also, analyze this product and figure out what customer demographic is this product for. Based on these things, come up with a detailed launch strategy for the product. List the TAM vs SAM vs SOM. TAM or Total Available Market is the total market demand for a product or service. SAM or Serviceable Available Market is the segment of the TAM targeted by your products and services which is within your geographical reach. SOM or Serviceable Obtainable Market is the portion of SAM that you can capture.""",
    """Acceptance Criteria:
Define the quality of completeness required to be able to get to the MVP stage of this product.""",
    """Technical Feasibilities:
Outline the technical roadmap for this product. What mobile devices should this application be available for? What is a scalable and reliable tech stack which can be used for the frontend and the backend for this application?""",
    """Timeline:
Define the timeline for the product development. In addition to the timeline, what are the resources required to complete this project. Think about the resources required for each stage of the project, the number of employees required for each stage, and the time required for each stage."""
]



def generate_chat_prd_gpt(product_description, product_name='Not mentioned'):

    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        max_retries=6,
    )

    wandb.init(
        project="chat-prd-gpt-4-v1.4",
        config={
            "model": "gpt-4",
            "temperature": 0
        },
        entity="arihantsheth",
        name=f"{product_name}_gpt-4",
    )

    memory = ConversationBufferMemory()

    chain = LLMChain(
        llm=chat,
        memory=memory,
        prompt=prompt_template,
        verbose=False
    )

    with get_openai_callback() as callback_init:
        initial_output = chain.predict(
            input=f"""\
I want to create the following new product:
{product_name}. 
If the product name is not mentioned, generate a product name.

Product description: {product_description}

DO NOT START WRITING. WAIT FOR THE HUMAN TO WRITE "Start generating the PRD" BEFORE YOU START WRITING.
""",
            callbacks=[WandbTracer()]
        )
        output = ""

        for i, prompt in enumerate(prompts_list):
            output += chain.predict(
                input=prompt,
                callbacks=[WandbTracer()]
            )

            output += "\n\n"
            print(f"Prompt {i+1} of {len(prompts_list)}")

    prd_cost += callback_init.total_cost
    prd_prompt_tokens += callback_init.prompt_tokens
    prd_completion_tokens += callback_init.completion_tokens

