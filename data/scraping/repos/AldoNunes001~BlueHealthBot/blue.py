import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM_MSG = [
    {
        "role": "system",
        "content": "Você é o Blue, um assistente virtual para ajudar homens a manterem-se saudáveis de maneira geral, especialmente quanto ao câncer de Próstata.",
    }
]

st.set_page_config(page_title="Buddy Bot", page_icon="🤖")

st.title("🤖 Blue Health Bot")
st.subheader("Como posso te ajudar?")
# st.markdown("""
# <h1 style='display: inline; font-size: 32px;'>🤖 Buddy Bot</h1>
# <span style='font-size: 20px; margin-left: 15px;'>Você não está sozinho(a)!</span>
# """, unsafe_allow_html=True)

# with st.expander("Aviso Legal"):
#     st.write(
#         """
#         O Buddy Bot foi desenvolvido para fornecer uma interface interativa que \
#         responde e fornece suporte em situações onde o usuário pode necessitar de \
#         companhia para conversar. O Buddy Bot pode oferecer respostas automáticas com \
#         a intenção de ajudar a proporcionar algum conforto ou alívio temporário.

#         Por favor, esteja ciente de que:

#         O Buddy Bot não é um profissional de saúde mental licenciado, nem um \
#         conselheiro, psicólogo ou psiquiatra. Ele não fornece conselhos médicos, \
#         diagnósticos ou tratamentos.
#         As respostas fornecidas pelo Buddy Bot não devem ser usadas como um substituto \
#         para o aconselhamento profissional. Se você está passando por uma crise ou se \
#         você ou outra pessoa estiver em perigo, entre em contato com um profissional de\
#         saúde mental, uma autoridade competente, ou ligue para o Centro de Valorização \
#         da Vida (CVV) no número 188, que fornece apoio emocional 24/7, ou acesse o site\
#         https://www.cvv.org.br/.
#         O Buddy Bot não tem a capacidade de interpretar situações de crise, emergências\
#         médicas ou de saúde mental, ou de fornecer ajuda em tempo real.
#         Todas as interações com o Buddy Bot são baseadas em inteligência artificial, o \
#         que significa que as respostas são geradas automaticamente e não são \
#         monitoradas por seres humanos em tempo real.
#         Respeitamos sua privacidade. Todas as conversas com o Buddy Bot são anônimas e \
#         não coletamos, armazenamos ou compartilhamos quaisquer dados pessoais do \
#         usuário. Nosso objetivo é proporcionar um espaço seguro para você se expressar.
#         Ao utilizar o Buddy Bot, você concorda com este Aviso Legal e compreende que \
#         qualquer ação ou decisão tomada com base nas respostas do Buddy Bot é de sua \
#         responsabilidade total.
# """
#     )

# Set OpenAI API key from Streamlit secrets


# Set a default mode
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-1106-preview"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun (when we hit enter)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Diga qual sua dúvida ou preocupação"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        responses = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=SYSTEM_MSG
            + [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages
            ],
            stream=True,
        )

        for response in responses:
            full_response += response.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
