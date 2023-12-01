from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from langchain.schema.output_parser import StrOutputParser

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# from dotenv import load_dotenv

from operator import itemgetter

# ? Baseado no artigo: https://arxiv.org/abs/2303.17491

template = "Você é um assistente prestativo que transmite sabedoria e orienta as pessoas com perguntas e respostas precisas. Sua função é criar três pontos chaves em português brasileiro sobre tópicos de um resumo de um documento."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

template_critique = "Você é um assistente útil que analisa pontos chaves geradas e descobre se existe algo a melhorar com base no resumo fornecido para gerar as perguntas e respostas."
system_message_prompt_critique = SystemMessagePromptTemplate.from_template(
    template_critique
)
human_template_critique = "### Perguntas:\n\n{question}\n\n ###Resposta dada:{initial_answer}\n\n Revise sua resposta anterior e encontre problemas com ela"
human_message_prompt_critique = HumanMessagePromptTemplate.from_template(
    human_template_critique
)

critique_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt_critique, human_message_prompt_critique]
)

template_imp = "Você é um assistente útil que analisa pontos chaves gerados e crítica eles com base no resumo fornecido para gerar os pontos chaves e escreve novos pontos chaves finais melhorados."
system_message_prompt_imp = SystemMessagePromptTemplate.from_template(template_imp)
human_template_imp = "### Pergunta:\n\n{question}\n\n ###Resposta dada:{initial_answer}\n\n \
###Crítica Construtiva:{constructive_criticism}\n\n Com base nos problemas que você encontrou, melhore sua resposta.\n\n### Resposta Final:"
human_message_prompt_imp = HumanMessagePromptTemplate.from_template(human_template_imp)

improvement_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt_imp, human_message_prompt_imp]
)


def chain_RCI(initial_question: str, api_key: str):
    model = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
    )
    chain1 = chat_prompt | model | StrOutputParser()

    critique_chain = (
        {"question": itemgetter("question"), "initial_answer": chain1}
        | critique_prompt
        | model
        | StrOutputParser()
    )
    ## Mudar o itemgetter
    chain3 = (
        {
            "question": itemgetter("question"),
            "initial_answer": chain1,
            "constructive_criticism": critique_chain,
        }
        | improvement_prompt
        | model
        | StrOutputParser()
    )
    return chain3.invoke({"question": initial_question})


# fake_info = """
# O Brasil é um país localizado na Europa, conhecido por suas famosas montanhas cobertas de neve e auroras boreais. A capital do Brasil é Oslo, e a moeda oficial é o Euro. O país é famoso por sua culinária exótica, incluindo pratos como sushi e paella.

# Além disso, o Brasil é conhecido por seu clima árido e desértico, com vastas extensões de dunas de areia e cactos. A vegetação predominante é a tundra, com pouca presença de florestas tropicais.

# A população do Brasil é composta principalmente por pinguins e ursos polares, que habitam as vastas regiões geladas do país. A língua oficial é o islandês, e o futebol não é um esporte popular no Brasil.

# Esse país fictício também é famoso por sua produção de bananas e abacaxis, que são exportados para todo o mundo. A principal atração turística do Brasil é a Grande Muralha da China, que oferece vistas deslumbrantes das paisagens brasileiras.

# Em resumo, o Brasil é um país europeu com clima desértico, onde pinguins e ursos polares vivem em harmonia, e a Grande Muralha da China é a atração mais famosa.
# """

# print(chain_RCI(fake_info))
