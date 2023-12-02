from langchain.chains import LLMChain
from src.dorahLLM.maritalkllm import MariTalkLLM
from langchain.prompts import PromptTemplate
from src.dorahLLM.browserless_api import get_text_sites
from src.dorahSearch.wikipedia_api import get_sumary

from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools, initialize_agent, AgentType

from src.dorahSearch.google_api import get_links, _google_search
from src.dorahSearch.wikipedia_api import _wikipedia_search
from src.dorahLLM.maritalk_topics import generate_topics_from_text

from random import randint

def summary_text(input_subject: str, input_text: str) -> str:
    template = """Você faz um resumo do texto sobre {subject}

Texto sobre Ração animal: ""Ração animal é o alimento dado para animais, tais como gado e animais de estimação.
Algumas rações proveem de uma dieta saudável e nutritiva, enquanto outras carecem de nutrientes.
Existe uma grande variedade de rações para cães, mas os dois tipos principais são a ração composta e a forragem.
A ração é indicado como o principal alimento para os animais.
Produção brasileira
A evolução do setor de alimentação animal acompanha, impulsiona e reflete em outros setores da economia,
caracterizando-se como um importante elo dentro da agroindústria brasileira.
Em 2011, o setor de alimentação animal consumiu 35% da produção nacional de farelo de soja e
quase 60% da produção nacional de milho, sendo que para este último há projeções de consumo de 60 milhões de toneladas
para 2020. [1] Além do envolvimento com mercado de grãos e outras matérias-primas, movimenta ainda a indústria
química de produção de insumos, vitaminas e minerais, e a indústria alimentícia humana, por integrar a principal
fonte de produção de proteína animal destinada ao consumo humano.
Impulsionada pelo crescimento da população e incremento no fornecimento de alimentos seguros, evidencia-se a
importância da tecnologia de rações.
Tipos de Rações
As rações para animais podem ser classificadas em vários tipos, dependendo do teor de humidade, qualidade da ração,
pela indicação para um estadio de vida, ou pelos ingredientes utilizados.[2]
As rações classificam-se pelo teor de humidade em rações secas, com teores até 10%, e rações húmidas,
com teores até 70%. Todas as rações que se apresentem no rótulo como "completas e equilibradas" deverão proporcionar
uma nutrição equilibrada.
""

Resumo:A alimentação animal é voltada para animais como gado e animais de estimação. Esse setor tem grande importância e reflete em outros setores da economia. Existe uma grande variedade de alimentos, mas os dois tipos principais são alimentos compostos e forragem. As rações podem ser classificadas em vários tipos, por exemplo pelo teor de umidade em rações secas e rações úmidas.

Texto sobre Geografia:""Geografia é, nos dias atuais, a ciência que estuda o espaço geográfico, produzido por meio da dinâmica das relações estabelecidas entre o homem e o meio. Em suma, a Geografia analisa a dinamicidade das relações entre a sociedade e a natureza, capazes de transformar o espaço geográfico. A maneira como essas relações são estabelecidas confere à Geografia sua identidade e importância.

O estudo das dinâmicas estabelecidas no espaço geográfico permite compreender a organização do espaço terrestre, contribuindo para que a sociedade alcance meios de explorar e transformar o meio ambiente sem agredi-lo. Dessa forma, desenvolvem-se alternativas para melhorar as relações socioespaciais.

É válido lembrar, contudo, que essa definição não é unânime entre os geógrafos, pois a Geografia, enquanto ciência, sofreu diversas transformações ao longo dos anos. Portanto, não há como afirmar que haja um consenso entre estudiosos a respeito do objeto de estudo e da orientação metodológica dessa área.
O que a Geografia estuda?
Geografia é a ciência que estuda as relações sociais estabelecidas no espaço geográfico, ou seja, as relações entre a sociedade e o meio. Esse espaço é transformado pelo homem e está, por isso, em constante modificação. Contudo, é difícil limitar o que é estudado pela Geografia ou não, visto que essa é uma ciência horizontal, ou seja, seu campo de estudo é amplo e relaciona-se com outras ciências, transcendendo seu próprio saber.

Assim, a Geografia, em virtude de sua orientação, é diferenciada dos demais saberes científicos. Trata-se de um estudo categorial, que abrange conceitos que definem sua orientação, como lugar, paisagem, território e região.
O que significa Geografia?
A palavra “geografia” tem origem grega e é formada pelos radicais “geo”, que significa Terra, e “grafia”, que significa descrição. Essa nomenclatura refere-se à definição antiga da ciência geográfica, que relacionava Geografia somente aos fenômenos que ocorrem na superfície terrestre.
Ramos da Geografia
A Geografia é dividida em alguns ramos, o que não significa que essa ciência deve ser estudada de forma compartimentada. Essa divisão é feita apenas para nortear os estudos, visto que as relações entre o meio e a natureza são indissociáveis.

As duas frentes principais da Geografia são:

1. Geografia Geral
* Geografia Humana: estuda a interação entre a sociedade e o espaço, envolvendo aspectos políticos, socioeconômicos e culturais. A Geografia Humana divide-se em categorias, como Geografia Urbana, Geografia Rural e Geografia Econômica.
* Geografia Física: estuda a dinâmica da Terra e dos fenômenos que ocorrem na superfície terrestre. A Geografia Física divide-se em categorias, como Climatologia, Geomorfologia, Geografia Ambiental e Hidrologia.

2. Geografia Regional

A Geografia Regional estuda as regiões da Terra de forma descritiva, a fim de entender as características e particularidades de cada uma delas.
""

Resumo: Geografia é, nos dias atuais, a ciência que estuda o espaço geográfico, produzido por meio da dinâmica das relações estabelecidas entre o homem e o meio. Esse estudo permite compreender a organização do espaço terrestre, contribuindo para que a sociedade alcance meios de explorar e transformar o meio ambiente sem agredi-lo. É válido lembrar, contudo, que essa definição não é unânime entre os geógrafos, pois a Geografia, enquanto ciência, sofreu diversas transformações ao longo dos anos. Essa nomenclatura refere-se à definição antiga da ciência geográfica, que relacionava Geografia somente aos fenômenos que ocorrem na superfície terrestre. A Geografia é dividida em alguns ramos, as duas frentes principais são a Geografia Geral, que inclui a Geografia Humana e Física, e a Geografia Regional.

Texto sobre Dia das Crianças: ""O Dia das Crianças é uma data comemorativa que alude à importância dos direitos da criança e à luta contra o abuso infantil. Quando ouvimos falar em Dia das Crianças, a imagem que nos vem a cabeça é sempre uma: presentes. Isso, é claro, não poderia deixar de ser, pois quem não gosta de presentes? No entanto, a celebração do Dia das Crianças não tem o intuito de apenas presentear os nossos pequenos. Na verdade, essa data é bastante significativa para o que realmente a criança representa. 
Origem dos Dias Internacional e Universal da Criança
Oficialmente, uma das primeiras convenções sobre uma data comemorativa internacional em homenagem à criança aconteceu em 1925, durante a Conferência Mundial pelo bem-estar da criança, realizada em Genebra, Suíça. Nessa ocasião, o dia 1º de junho ficou marcado como o Dia Internacional da Criança. No ano anterior, 1924, a então chamada "Liga das Nações" fundou a "Declaração dos Direitos da Criança" para fundamentar os cuidados especiais que deveriam ser tomados em relação a todas as crianças diante da fragilidade do ser humano em sua infância. Dessa medida surgiram atos legais que proibiram o trabalho infantil e a violência contra a criança.

Leia também: Quando a criança não deve ir à escola?

Tempos depois, em 1954, durante a Assembleia Geral das Nações Unidas, o dia 20 de Novembro foi estabelecido como o Dia Universal da Criança. O objetivo era encorajar os demais países a estabelecerem uma data para promover ações que garantiriam direitos e o bem-estar da criança. Em 1959, a Assembleia Geral das Nações Unidas adotou a "Declaração dos Direitos da Criança", com algumas modificações, e cada país passou a estabelecer uma data comemorativa para celebrar os direitos da criança.

Dia das Crianças no Brasil
No Brasil, entretanto, a data já havia sido estipulada ainda na década de 1920. O deputado federal do Rio de Janeiro, Galdino do Valle Filho, conseguiu a aprovação da lei, em 1924, que instituía o dia 12 de outubro como o Dia da Criança.

Veja também: Estatuto da Criança e do Adolescente

Todavia, essa data passaria despercebida até a década de 1950, quando houve uma campanha de marketing da empresa de brinquedos Estrela. A fabricante de brinquedos usou a data para promover sua linha de bonecas de nome "Bebê Robusto". Anos depois, a data foi mais uma vez reforçada pela campanha publicitária da empresa de produtos de higiene infantil Johnson & Johnson. A empresa lançou a campanha "Bebê Johnson", que teve sua primeira edição em 1965 e acabou se tornando o concurso de beleza infantil mais conhecido no país.
""

Resumo: O Dia das Crianças é uma data comemorativa que alude à importância dos direitos da criança e à luta contra o abuso infantil. Durante a Conferência Mundial pelo bem-estar da criança, o dia 1º de junho ficou marcado como o Dia Internacional da Criança. Entretanto, no Brasil a data já havia sido estipulada ainda na década de 1920, mas essa data passaria despercebida até a década de 1950, quando houve uma campanhas de empresas com produtos infantis.

Texto sobre {subject}:""{input}
""

Resumo:"""

    prompt = PromptTemplate.from_template(template)
    model = MariTalkLLM()
    chain = LLMChain(prompt=prompt, llm=model)
    output_date = chain(inputs={"subject": input_subject, "input": input_text})
    output = output_date["text"]
    return output


def summary_sites(term: str, llm_interface, load_interface, links: list[str], wiki_interface) -> str:
    summaries = get_sumary(term, wiki_interface)

    try:
        list_doc = load_interface(links)
        for text_date in list_doc:
            length_text = len(text_date.page_content)
            if length_text < 18:
                continue
            page_init = int(float(length_text) * 0.0560)  # evitar cabeçalho
            page_end = 8000 + page_init  # impede limite de tokens da Maritalk
            if length_text < page_end:
                page_end = length_text
            partial_summary = llm_interface(term, text_date.page_content[page_init:page_end])
            summaries += partial_summary
    except (ValueError, TypeError):
        # Impedir que erro do BrowserlessLoader atrapalhe se já acessou o wikipedia
        pass

    if summaries == "Summary Not Found :(":
        return ''
    final_summary = llm_interface(term, summaries)
    return final_summary


def perform_topics(topic):
    one_term = topic
    urls = get_links(one_term, _google_search)

    url = urls[0]

    one_summary = summary_sites(one_term, summary_text, get_text_sites, url, _wikipedia_search)
    one_topics = generate_topics_from_text(one_summary)
    
    return one_topics


def summary_text_test(input_subject: str, input_text: str) -> str:
    if input_subject == "" or input_subject == "assunto não especificado" or input_text == "Texto incorente":
        return ''

    responses = ["Final Answer: A Independência do Brasil foi o processo histórico de separação entre o então Reino do Brasil e o Reino de Portugal e Algarves, que ocorreu no período de 1821 a 1825, colocando em violenta oposição as duas partes (pessoas a favor e contra). As Cortes Gerais e Extraordinárias da Nação Portuguesa, instaladas em 1820, como consequência da Revolução Liberal do Porto, tomaram decisões que tinham como objetivo reduzir a autonomia adquirida pelo Brasil. O processo de independência foi liderado por Dom Pedro I, que se tornou o primeiro imperador do Brasil. A proclamação foi realizada no dia 7 de setembro e foi seguida por um período de transição, com a formação de um governo provisório e a convocação de uma Assembleia Constituinte. Durante esse período, ocorreram conflitos entre os partidários de Dom Pedro I e os que defendiam uma maior autonomia das províncias. A independência foi finalmente reconhecida por Portugal em 1825, após a assinatura de um tratado de paz."]
    llm = FakeListLLM(responses=responses)
    tools = load_tools(["python_repl"])
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    return agent.run("Você faz um resumo do texto")

if __name__ == "__main__":
    one_term = "Brasil"
    urls = get_links(one_term, _google_search)
    one_summary = summary_sites(one_term, summary_text, get_text_sites, urls, _wikipedia_search)
    print(f"Resumo: \n{one_summary}\n")
    one_topics = generate_topics_from_text(one_summary)
    print(f"Tópicos: \n{one_topics}")
