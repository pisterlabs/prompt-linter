from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from configs import conf


def run():
    openAI = OpenAI(
        temperature=0.2,
        openai_api_key=conf.get("api_key"),
    )

    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [human_message_prompt])

    chain = LLMChain(
        llm=openAI,
        prompt=chat_prompt_template,
    )

    resp = chain.run("colorful socks")

    print(resp)
