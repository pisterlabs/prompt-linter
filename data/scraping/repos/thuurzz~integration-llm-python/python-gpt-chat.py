from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

openai_api_key="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"

def handler():
    mensagem = "Você conhece a Porto Seguro? Me fale sobre essa empresa."
        
    pergunta_cliente = mensagem
    
    print(pergunta_cliente)

    template = """Pergunta: {pergunta_cliente}"""

    prompt = PromptTemplate(template=template, input_variables=["pergunta_cliente"])

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    resp = llm_chain. run(pergunta_cliente)

    print(resp)

handler()
