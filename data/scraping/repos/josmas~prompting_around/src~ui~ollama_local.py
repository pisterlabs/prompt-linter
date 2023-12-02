from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama
from langchain.schema import AIMessage
import tiktoken

# TODO : I also need a decorator to send the data to the DB
def generate_text(model: str, temperature: float, prompt_input: str) -> (AIMessage, dict):
    prompt = ChatPromptTemplate.from_template("tell me a joke about {prompt_input}")

    model = ChatOllama(model=model,
               temperature=temperature,
               verbose=False,)
    chain = prompt | model

    response: AIMessage = chain.invoke({"prompt_input": prompt_input})

    encoding = tiktoken.get_encoding("cl100k_base")
    input = prompt.format(prompt_input=prompt_input)
    input_tokens = len(encoding.encode(input))
    output_tokens = len(encoding.encode(response.content))
    total_num_tokens = input_tokens + output_tokens

    return response, {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_num_tokens": total_num_tokens}
