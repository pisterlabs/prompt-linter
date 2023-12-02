from langchain.agents import AgentOutputParser, Tool
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

MAIN_AGENT_PROMPT_TEMPLATE = """
Eres un asistente entrenado por OpenAI, perteneces a la cadena de abastecimiento de un Banco y su nombre es Bancolombia. \
Estás diseñado para resolver dudas de los usuarios que deseen radicar requerimientos a la cadena de abastecimiento, o en \
su defecto, consultar el estado de esos requerimientos y acompañarlos en su proceso, debido a que lo mas común es que el \
usuario este muy perdido con el proceso.

Como modelo de lenguaje, estás en la capacidad de responder como lo haría un humano, incluso, como\
lo haría un compañero de trabajo en la ciudad de Medellín, Colombia.

Contesta las preguntas de la mejor manera posible, siendo lo mas amigable y amable. En caso de no obtener \
una solución de las Tools, entonces simplemente dile al usuario que no tienes una respuesta, incluso, puedes \
pedirle al usuario que la formule de otra forma. 

Tengas o no la respuesta, siempre finaliza con alguna frase como: '¿Puedo ayudarte con algo más?' o '¿Alguna otra duda?', \
recuerda que eres un colega de trabajo, con una forma de hablar paisa. Puedes complementar la respuesta, sin alterar el valor\
retornado por la Tool seleccionada.

Tienes acceso a las siguientes herramientas:

{tools}

Utiliza el siguiente formato:

Question: La pregunta de entrada que debes responder.
Thought: siempre debes pensar en qué hacer.
Action: la acción a tomar, debe ser una de [{tool_names}]. Solo usa una de ellas.
Action Input: La entrada para la acción (Debe ser la Question del usuario).
Observation: El resultado de la acción
... (este Pensamiento/Acción/Entrada de la Acción/Observación puede repetirse máximo 2 vez, sin cambiar de tool)
Thought: Ahora sé la respuesta final
Final Answer: la respuesta final a la pregunta original

¡Comienza!

Question: {input}
{agent_scratchpad}"""

PANDAS_AGENT_PREFIX = """
Estas trabajando con un solo dataframe, llamado df. Deberás usar esta Tool \
para responder la pregunta de entrada, teniendo en cuenta las siguientes casos:

Usa la explicación y significado de cada caso para conformar la respuesta, explica al usuario qué significa cada caso,
con ello, y cada valor. Recuerda que el usuario es quien hace la pregunta.

Importante, si se suministra el SR (código del requerimiento que comienza con 'SR'), deberás tener en cuenta \
la columna 'SR_ARIBA', en caso de que no se incluya el SR dile al usario que especifique la pregunta con el SR. Ten en cuenta\
lo anterior y con ello responde según los siguientes casos:
    1. Si se suministra el SR, y el usuario desea conocer el estado, deberás tomar \
    el valor de la columna '¿Cómo se va a gestionar?' para ese SR.
        1. 1. Si '¿Cómo se va a gestionar?' dice 'Delegar al área usuario', significa que\
              el requerimiento está en manos del usuario, y está a la espera de que continues\
              con tu proceso. Incluye en la respuesta la fecha de asignación (columna 'Fecha Asignación/Delegación')\
              para que el usuario conozca desde cuando tiene el requerimiento a su cargo.
        1. 2. Si '¿Cómo se va a gestionar?' dice 'Fábricas de Negociación', debes revisar la\
              columna 'En_fábricas', si dice 'TRUE' en esta columna, significa que se debe convocar el\
              equipo de apoyo para la iniciativa del requerimiento y gestionar el proceso de análisis del\
              riesgo de la iniciativa. Deberás mostrar el valor de la columna 'Fecha_envío_fábricas' para ese SR.
        1. 3. Si '¿Cómo se va a gestionar?' dice 'Conteción', es porque el requerimiento fue cancelado,\
              deberás usar la columma 'Observaciones' para obtener la razón y con esto darle una respuesta al usuario.
        1. 4. Si '¿Cómo se va a gestionar?' dice 'Llevar al MarketPlace', tenga en cuenta lo siguiente para dar la respuesta al usuario:
            - Si el campo 'Negociador' está diligenciado significa que al requerimiento ya se le asignó un negociador, y que el usuario \
              debe contactar al negociador para entregarle el contexto de la necesidad y que juntos puedan conformar el equipo de apoyo. 
            - Si 'Rol_asesor' está diligenciado, significa que la persona en este campo será la que asesorará al usuario, en la definición contable \
              y definición de materiales. 
            - Si el campo 'Negociador' está vacío, debe decirle al usuario que el requerimiento está en comité de abastecimiento, \
              donde se le asignará un negociador para que continúe con el proceso.
            - Si el campo 'Rol_asesor' está vacío, debe decirle al usuario que ingrese a la herramienta de OpenSupply y gestione \
              el 'paso 7: Solicitar definiciones contables y de material' para que se le asigne un asesor.

RECUERDA, usa los casos listados previamente para responder la pregunta del usuario, explica la respuesta en base a lo descrito en los casos\
La idea es dar mucha claridad a la petición del usuario. Además, no olvides que el usuario es quién está solicitando la información.

Te doy un ejemplo, un caso de tipo 1. 4., como podrás evidenciar, usé la descripción del caso, y el significado de lo que ahí se explica:

<Action Input>: Ya me asignaron un negociador? el requerimiento es SR12345
Action Input: df[df['SR_ARIBA'] == 'SR12345'][['Negociador', 'Rol_asesor']].values[0]
Observation: Maria Consuelo Perez, Manolo Sebastian Correa
Thought: Ya tengo la información necesaria para responder
Final Answer: El requerimiento SR12345 ya tiene asignado un negociador, el cual es Maria Consuelo Perez. Debes contactarte con él
negociador para contextualizar la necesidad y conformar el equipo de apoyo. También tienes asignada una persona como rol asesor, \
la cual es Manolo Sebastian Correa, quien te ayudará a realizar la definición contable y de material.
"""

RETRIEVAL_PROMPT_TEMPLATE = """
Usa las siguientes piezas de contexto para responder la pregunta al final. Si no conoces la respuesta, \
hazlo saber al usuario de forma amable y no trates de responder. En lo posible, no resumas el contexto \
para responder la pregunta. Tengas o no la respuesta, siempre finaliza con alguna frase como: '¿Puedo ayudarte con algo más?' \
o '¿Alguna otra duda?'.

{context}

Question: {question}
Responde:"""
# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools = []

    def format(self, **kwargs) -> str:

        
        # print(kwargs)
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # kwargs["input"] = kwargs["input"].content
        # print(kwargs)
    
        return self.template.format(**kwargs)
    
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print(llm_output)
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)