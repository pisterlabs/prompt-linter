from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessage, HumanMessagePromptTemplate


def generate_prompts(user_input=None):
    context_01 = """ in a clear and concise manner to a child in one sentence. """
    context_02 = """ with a focus on scientific details explain to a scientist """

    prompt = PromptTemplate.from_template(
        "Respond to the user's question that is delimited by triple backticks "
        + "to provide some context {context}"
        + "text: ```{user_input}``"
    )

    prompt_01 = prompt.format(user_input=user_input, context=context_01)
    prompt_02 = prompt.format(user_input=user_input, context=context_02)

    return [prompt_01, prompt_02]

def generate_chat_prompts(user_input=None):
    
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant that translates English to Portuguese."
                )
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    
    prompt_01 = [SystemMessage(content='You are a helpful assistant that translates English to French.'), HumanMessage(content='Translate this sentence from English to French. I love programming.')]
    prompt_02 = template.format_messages(text=user_input)
    

    return [prompt_01, prompt_02]
