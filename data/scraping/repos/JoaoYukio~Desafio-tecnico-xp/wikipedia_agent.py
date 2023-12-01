from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType


def search_wikipedia(query: str):
    """
    Pesquise um tópico na Wikipedia e retorne o resumo.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    result = wikipedia.run(query)
    if result:
        return result.split("\n")[1]
    else:
        return "Nenhum documento encontrado."


def lookup(llm, num_perguntas: int, text: str) -> str:
    """
    Pesquise um tópico na Wikipedia e retorne o primeiro documento.
    """
    template = """
    Gostaria que criasse {num_perguntas} pontos interessantes em português brasileiro sobre o texto {text} que você acabou de ler.
    Esses pontos interessantes devem ser formatados como uma lista numerada.
    Não deve ter nada além dos pontos interessantes formatados na sua resposta. 
    """

    tools_for_agent = [
        Tool(
            name="Procurar na Wikipedia",
            func=search_wikipedia,
            description="Procura um tópico na Wikipedia e retorna o resumo.",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(
        template=template, input_variables=["num_perguntas", "text"]
    )

    questions = agent.run(
        prompt_template.format_prompt(
            num_perguntas=num_perguntas,
            text=text,
        )
    )

    return questions


"""
Exemplo de output do agente:

> Entering new AgentExecutor chain...
 I need to find information about the War of Canudos
Action: Procurar na Wikipedia
Action Input: Guerra dos Canudos
Observation: Summary: The War of Canudos (Portuguese: Guerra de Canudos, Portuguese pronunciation: [ˈɡɛʁɐ dʒi kɐˈnudus], 1896–1898) was a conflict between the First Brazilian Republic and the residents of Canudos in the northeastern state of Bahia. It was waged in the aftermath of the abolition of slavery in Brazil (1888) and the overthrow of the monarchy (1889). The conflict arose from a millenarian cult led by Antônio Conselheiro, who began attracting attention around 1874 by preaching spiritual salvation to the poor population of the sertão, a region which suffered from severe droughts. Conselheiro and his followers came into atrittion with the local authorities after founding the village of Canudos. The situation soon escalated, with Bahia's government requesting assistance from the federal government, who sent military expeditions against the settlement.
Thought: I now have enough information to create 3 interesting points
Final Answer:
1. A Guerra dos Canudos foi um conflito entre a Primeira República Brasileira e os habitantes de Canudos, no nordeste do estado da Bahia.
2. O conflito surgiu de um culto milenar liderado por Antônio Conselheiro, que começou a atrair atenção a partir de 1874, pregando a salvação espiritual para a população pobre do sertão, uma região que sofria com severas secas.
3. O governo da Bahia pediu ajuda ao governo federal, que enviou expedições militares contra o assentamento.

"""
