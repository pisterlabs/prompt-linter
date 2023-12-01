import os
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.tools import Tool
from tools import tools
from langchain.chains import LLMMathChain

st.session_state.disabled = False

with st.sidebar:
    if ('OPENAI_API_KEY' in st.secrets) and ('openai_model' in st.secrets):
        api_key = st.secrets['OPENAI_API_KEY']
        id_model = st.secrets['openai_model']
    else:    
        api_key = st.text_input("OpenAI API Key", placeholder='Ingresa tu OPEN API Key', type="password", disabled=st.session_state.disabled)
        id_model = st.selectbox('Modelo', ('gpt-3.5-turbo', 'gpt-3.5-turbo-1106', 'gpt-4', 'gpt-4-1106-preview'), index=None, placeholder='Selecciona un modelo', disabled=st.session_state.disabled)
    
    placeholder = st.empty()    
    with placeholder.container():
        if (not api_key) or (id_model is None):
            st.warning('Por favor, ingresa tus credenciales y selecciona el modelo!', icon='⚠️')
        else:
            os.environ['OPENAI_API_KEY'] = api_key
            st.session_state.disabled = True
            st.success('¡API KEY ingresada! \n\nYa puedes ingresar los mensajes. \n\n Para seleccionar otro modelo, refresca la página', icon='👉')

st.title("🔎 TaxBot")
st.write('Este es un chatbot de prueba para trabajar en relación al cálculo de impuestos en Chile. Por favor, ingresa tu pregunta en la casilla de más abajo.')


if (not api_key) or (id_model is None):
    st.info("Por favor, ingresa tus credenciales y selecciona el modelo!")
else:
    llm_m = OpenAI(temperature=0, model_name="gpt-3.5-turbo-1106", streaming=True)
    llm_math_chain = LLMMathChain.from_llm(llm_m)
    tools = tools + [Tool(
            name="Calculadora",
            func=llm_math_chain.run,
            description=" útil para responder preguntas matemáticas básicas y realizar sumas necesarias, como por ej, el cálculo del ingreso anual total.",
        )]

    descripcion_tools = ''
    for t in tools:
        descripcion_tools += '>'+t.name+': '+t.description+'\n'
        
    system_message=f'''Eres un chatbot especializado y diseñado exclusivamente para calcular impuestos en Chile. Todas los cálculos deben realizarse en Pesos Chilenos (CLP). Tu función principal es responder preguntas relacionadas con el cálculo de impuestos y guiar al usuario para obtener información detallada sobre sus diferentes tipos de ingresos con el objetivo de determinar su impuesto anual total. 
Para lograr esto, debes preguntar al usuario acerca de las distintas fuentes de ingreso que podría tener, que incluyen:

>Ingresos por sueldos o salarios recibidos.
>Ingresos por arriendo. DEBES PREGUNTAR si el ingreso por arriendo es DFL2. Si es DFL2, no se paga impuesto, por tanto no sumes este ingreso. 
>Ingresos por boletas de honorarios. DEBES identificar si es bruto o líquido. NO ASUMAS EL TIPO. Si es líquido, debes dividir el monto por 0.87.
>Ingresos como dueño o socio de una empresa.
>Créditos Hipotecarios.
>Total Créditos Tributarios. El total es la suma de:
    -Crédito por impuesto de ingresos brutos de boletas de honorarios.
    -Crédito por impuesto retenido en sueldos y salarios (impuesto único). 
    -Otros declarados por el usuario.
>Año Fiscal. NO ASUMAS el año si el usuario no lo hizo explícito. En ese caso, DEBES preguntar.

Si alguna información de esta lista no es declarada, DEBES PREGUNTAR AL USUARIO por ella. Si el usuario no sabe la respuesta, DEBES guiarlo en encontrar la respuesta. PUEDES PROPONER ASUMIR UN VALOR POR DEFECTO, pero DEBES PREGUNTAR AL USUARIO si está de acuerdo.
Una vez que hayas recopilado TODA LA INFORMACIÓN NECESARIA, debes sumar todos los ingresos para obtener el total del ingreso ANUAL y calcular el impuesto usando este ingreso.
Finalmente, debes sumar todos los créditos tributarios identificados y restarlos del impuesto calculado. Luego debes proporcionar el total a pagar o a favor (devolución de impuestos).
NO USES decimales para separar los miles. Usa PUNTO para separar los decimales.

Información Importante:
Monto Líquido de una Boleta de Honorarios: el receptor del servicio retuvo el impuesto de la boleta.

Herramientas disponibles:
{descripcion_tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do, skip to Final Answer if you think no action is needed.
Action: the action to take, should be one of {str([t.name for t in tools]).replace("'", '')}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Chat history:
{{chat_history}}

Question: {{input}}
Thought:{{agent_scratchpad}}'''

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
    )    


    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        st.session_state.steps = {}

    avatars = {"human": "user", "ai": "assistant"}

    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)

    if prompt := st.chat_input(placeholder='Escribe tu pregunta aquí'):
        st.chat_message("user").write(prompt)

        if not api_key:
            st.info("Por favor, ingresa tus credenciales y selecciona el modelo!")
            st.stop()

        llm = ChatOpenAI(temperature=0.2,model_name=id_model, streaming=True)
        tax_agent = initialize_agent(tools=tools,
                                llm=llm, 
                                memory=memory,
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                handle_parsing_errors=True,
                                return_intermediate_steps=True)

        tax_agent.agent.llm_chain.prompt.input_variables = ['chat_history', 'input', 'agent_scratchpad']
        tax_agent.agent.llm_chain.prompt.template = system_message

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = tax_agent(prompt, callbacks=[st_cb])
            # tax_agent.early_stopping_method()
            st.write(response["output"])
            # st.write(response)
            st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]