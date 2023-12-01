# Biblioteka Langchain. Szablony oraz połączenie z OpenAI,

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain


if __name__ == '__main__':
    with open('api.key', 'r') as openai_api_key:
        openai_api_key = openai_api_key.read().strip()

    llm = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=openai_api_key
    )

    template = """Opinia użytkownika: {review}

    Oceń opinię ({options}):"""

    prompt_template = PromptTemplate(
    input_variables=["review", "options"],
    template=template
)

    review = "Jeśli  ktoś  lubi  zatłoczone hałaśliwe miejsca to może  w lipcu  się  tam wybrać, mi to zupełnie  nie odpowiada,  a wręcz jest męczące, ciężko  przemieszczać się po promenadzie a wejściówki  na tak zatłoczone  molo  dla  4 osobowej  rodziny  to i tak za drogie. "

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response = llm_chain.run(
        review=review,
        options="pozytywna/negatywna",
        max_tokens=50,
        temperature=0.0,
        stop="\n"
    )

    print(response)
