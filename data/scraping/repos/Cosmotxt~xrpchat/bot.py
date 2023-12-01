#openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

#pinecone
import pinecone
from langchain.vectorstores import Pinecone

#tools
from langchain.tools import Tool

#agente
from langchain.agents import initialize_agent

# retrieval
from langchain.chains import RetrievalQA

#news tool
from langchain.utilities import GoogleSerperAPIWrapper

#retriever
from langchain.chains import RetrievalQA

#memory_and_prompt
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#os
import os
from dotenv import load_dotenv

load_dotenv()

# setting keys and environments

# os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
pinecone_api_key = os.getenv('pinecone_api_key')
pinecone_environment = os.getenv('pinecone_environment')

# setting the llm
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm = ChatOpenAI(
    model_name='gpt-3.5-turbo-16k',
    temperature=0.5,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
)

# setting pinecone
embeddings = OpenAIEmbeddings()
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = 'indexvegacrypto'
vectorstore = Pinecone.from_existing_index(index_name, embeddings)

# setting memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True
)

# prompt template
system_template = """
Introducing: Vegabot, your friendly companion for navigating the world of the digital real!
Vegabot works on Vega Crypto and he is at your service, poised to help the user unravel the mysteries
surrounding the upcoming digital real. If you're seeking insights about this groundbreaking
financial evolution, you're in the right place. Feel at ease to inquire about anything related
to the digital real, and Vega Crypto's financial expertise.

Vegabot's expertise lies in elucidating queries related to the digital real and offering insights
into the realms of Vega Crypto and the broader financial landscape. The boundaries of its knowledge
are confined to matters of the digital real and the intricacies of finance, an assurance that keeps
our interactions informative and enlightening.

Should a question arise that falls beyond the scope of Vegabot's wisdom, transparency is its virtue.
Rest assured, it will openly acknowledge its limits. Yet, it remains committed to furnishing you with
the most comprehensive responses based on the information it possesses.

Embrace the opportunity! Pose your inquiries about the digital real, and Vegabot will go the extra mile
to present you with lucid and insightful answers, wrapped in an aura of friendliness and charisma.
"""

system_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_prompt = HumanMessagePromptTemplate.from_template("{input}")
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# retrieval tool
qa = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vectorstore.as_retriever(),
)

# retriever sources function

def sources(q):
  sources = vectorstore.similarity_search_with_relevance_scores(q)
  return [sources[0][0].metadata['source'], sources[1][0].metadata['source']]

# tradutor de entrada e saída

from langdetect import detect
from googletrans import Translator, LANGUAGES

def translator(text, lang):
  try:
    text_language = detect(text)
    if text_language == lang:
      return text

    text_translator = Translator()
    translated_text = text_translator.translate(text, src=text_language, dest=lang)
    return translated_text.text
  except Exception as e:
    return('Desculpe, não consigo responder sua pergunta')

# seting tools 

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description='Utilize the Knowledge Base tool to fetch answers directly from documents. All queries should looking for information using the Document search tool first.',
        return_direct=True
    ),
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=4,
    early_stopping_method='generate',
    memory=conversational_memory,
    return_direct=True
)

tips_responses = {
    'O que é o Real Digital e quais suas caracteristicas?': 
    '''
O Real Digital é uma moeda digital que está sendo desenvolvida pelo Banco Central do Brasil. É uma representação digital da moeda brasileira, o real, e estará disponível em uma plataforma eletrônica controlada pelo banco central.\n
**Aqui estão algumas características importantes do Digital Real:**
1. CBDC: Digital Real é um tipo de CBDC, o que significa que é uma moeda digital emitida por um banco central.Ele fornece confiabilidade, estabilidade e previsibilidade que vêm com regulamentação, semelhante à moeda física.
2. Tecnologia do Ledger distribuída (DLT): o Digital Real é construído com a tecnologia Distributed Ledger, especificamente uma rede descentralizada.Isso significa que as informações não são armazenadas em um único computador, mas em uma rede de computadores que verificam e fornecem acesso simultaneamente às informações, tornando o sistema mais seguro.
3. Contratos inteligentes: o Digital Real permite o uso de contratos inteligentes, que são contratos auto-executados com os termos do contrato diretamente escritos em linhas de código.Os contratos inteligentes permitem transações automatizadas e seguras, eliminando a necessidade de intermediários e reduzindo os custos.
4. Redução dos custos: Um dos benefícios do Real Digital é o potencial de redução de custos.Os contratos inteligentes e o uso de plataformas eletrônicas podem automatizar e simplificar as transações, tornando-as mais eficientes e mais baratas.
5. Acesso aos serviços financeiros tradicionais: o Digital Real pretende facilitar o acesso aos serviços financeiros tradicionais e permitir o desenvolvimento de novos modelos de negócios na plataforma DLT gerenciada pelo banco central.Isso pode potencialmente expandir a inclusão financeira e abrir novas oportunidades para indivíduos e empresas.
É importante observar que o Digital Real ainda está na fase de desenvolvimento, e sua implementação e impacto completos na economia ainda não foram completamente determinados.''',

    'O Real Digital é uma criptomoeda?': 
    '''
Não, o Real Digital não é uma criptomoeda. O Real Digital é uma moeda digital emitida pelo Banco Central do Brasil e é regulada pela autoridade monetária. 
Diferentemente das criptomoedas, o Real Digital não possui a mesma volatilidade de preços e é respaldado pelo Banco Central, oferecendo estabilidade 
e previsibilidade em seu valor. Além disso, o Real Digital tem o objetivo de facilitar o acesso a serviços financeiros tradicionais e 
reduzir custos de transação, enquanto as criptomoedas são tipicamente descentralizadas e não são controladas por uma autoridade central.''',

'O que é a Vega Crypto?':
'''
Vega Crypto é uma statup Brasileira de consultoria em web3 que oferece serviços de consultoria, pesquisa e educação financeira descentralizada para indivíduos e empresas.
    ''',

    'Qual a diferença do Real Digital para o pix?': 
    '''
# Diferenças entre Real Digital e PIX

O Real Digital e o PIX são duas iniciativas diferentes do Banco Central do Brasil, cada uma com seu próprio objetivo e características. 
Aqui estão as principais diferenças entre os dois:

1. Natureza e função:

- **Real Digital**: É a versão digital da moeda brasileira, o Real. O objetivo é modernizar o sistema monetário, permitindo transações com ativos 
digitais e contratos inteligentes em um ambiente seguro e regulamentado.
- **PIX**: É um sistema de pagamento instantâneo que permite transferências e pagamentos em tempo real, 24/7, sem a necessidade de intermediários. 
PIX não é uma moeda nova, mas uma maneira rápida e eficiente de mover o real existente.

2. Tecnologia:

- **Real Digital**: Opera em uma plataforma distribuída de tecnologia do Ledger (DLT), que fornece um sistema seguro e descentralizado para gravar transações.
- **PIX**: É baseado em uma infraestrutura de processamento centralizada, mas com operações quase instantâneas.

3. Regulamentação e emissão:

- **Real Digital**: Será emitido e regulamentado pelo Banco Central, servindo como uma extensão digital da moeda física.
- **PIX**: Não envolve a emissão de uma nova moeda. É simplesmente um meio de transferir e pagar usando o real existente.

4. Casos de uso:

- **Real Digital**: Pretende facilitar transações com ativos digitais e contratos inteligentes, além de outras aplicações financeiras avançadas, 
para otimizar e fornecer mais serviços financeiros aos brasileiros.
- **PIX**: Concentra-se em transferências e pagamentos rápidos, entre indivíduos, indivíduos e empresas ou entre empresas.

Em resumo, enquanto o Real Digital é uma representação digital da moeda brasileira com potencial para revolucionar o sistema financeiro e monetário, 
o PIX é simplesmente um sistema de pagamento que já transformou como os brasileiros transferem dinheiro e efetuam pagamentos em suas vidas diárias.''',

    'Quais os principais benefícios do Real Digital?': 
    '''
**Os principais benefícios do Real Digital incluem:**
* Facilitação do acesso a serviços financeiros tradicionais, especialmente para pessoas sem acesso a contas bancárias.
* Redução de custos de transação, tornando as transações mais eficientes e econômicas.
* Maior segurança e proteção contra fraudes, devido à tecnologia de criptografia utilizada.
* Possibilidade de uso de contratos inteligentes, que automatizam e garantem a execução de transações com base em condições pré-definidas.
* Potencial para impulsionar a inovação financeira e o desenvolvimento de novos modelos de negócios.
É importante ressaltar que o Real Digital está em fase de desenvolvimento e sua implementação completa ainda está por vir. Os benefícios mencionados são baseados nas expectativas e nas possibilidades que a moeda digital pode trazer.
    ''',

    'Quais os riscos do Real Digital?': 
    '''
**Os riscos do Real Digital podem incluir:**
* Possíveis vulnerabilidades de segurança que podem ser exploradas por hackers e cibercriminosos.
* Desafios técnicos e operacionais, como a escalabilidade da rede e a interoperabilidade com outros sistemas financeiros.
* Preocupações relacionadas à privacidade e proteção de dados, uma vez que as transações digitais podem envolver a coleta e o armazenamento de 
informações pessoais.
É fundamental que o Banco Central adote medidas robustas de segurança e privacidade para mitigar esses riscos e garantir a confiança dos usuários no Real Digital.

    '''
}

user_conversations = {}


def get_response(user_id, text, lang, messageId):
    if messageId == 1:
        if not tips_responses.__contains__(text):
           pass 
        else:
            response = tips_responses.get(text)
            return {'result': translator(response, lang), 'source': sources(text)}
    if user_id not in user_conversations:
        # Initialize conversation history for new user
        user_conversations[user_id] = []

    conversation_history = user_conversations[user_id]

    try:
        # Append user input to conversation history
        conversation_history.append(text)

        # Generate response using the conversation history
        output = agent(chat_prompt.format_prompt(input=text).to_string())['output']

        # Append agent response to conversation history
        conversation_history.append(output)

        response = {'result': translator(output, lang), 'source': sources(text)}
        return response
    except Exception as e:
        return {"error": str(e)}
    
    
    
