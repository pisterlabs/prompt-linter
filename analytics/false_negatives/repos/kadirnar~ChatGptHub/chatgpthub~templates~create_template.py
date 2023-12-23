def custom_prompt_template(
    model_name: str = "gpt-3.5-turbo",
    template: str = "You are a helpful assistant that English to Turkish and you are asked to translate the following text: {text}",
    input_variables: str = "text",
    text: str = "Hello, how are you?",
    temperature: float = 0.0,
):
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate

    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    prompt = PromptTemplate(
        input_variables=[input_variables], template=template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(text)
    return output
