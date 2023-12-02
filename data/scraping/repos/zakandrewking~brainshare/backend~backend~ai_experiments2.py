from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema import SystemMessage


async def chat_with_tools(query: str) -> tuple[str, int]:
    sysmsg = SystemMessage(content="You are a helpful assistant designed to output JSON.")
    prompt = sysmsg + HumanMessagePromptTemplate.from_template("{query}")
    model = ChatOpenAI(
        model_name="gpt-4-1106-preview",
        max_tokens=128,
    ).bind(
        response_format={
            "type": "json_object",
        }
    )
    chain = prompt | model
    res = chain.invoke({"query": query})
    return str(res.content), 0
