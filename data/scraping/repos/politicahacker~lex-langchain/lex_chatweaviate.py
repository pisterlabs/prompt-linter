import os

#LLM
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
#Memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

#CallBack
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#Weaviate Memory
from tools.uploadLib import Library

library = Library(os.getenv('WEAVIATE_URL'), os.getenv('WEAVIATE_API_KEY'))

from langchain.schema.messages import HumanMessage

class DynamicLibraryPromptTemplate(HumanMessagePromptTemplate):
    def validate_input_variables(cls, v):
        # Valide suas variáveis de entrada aqui
        return v

    def format(self, **kwargs) -> str:
        # Puxe o human_input do kwargs
        human_input = kwargs.get("human_input")

        # Obtenha informações da biblioteca com base no human_input
        sources = library.get_sources(human_input)

        # Formate 'sources' para incluí-los no prompt
        formatted_sources = self.format_sources(sources)

        # Crie o prompt final
        text = f"Esses são os trechos de documentos da nossa biblioteca.:\n{formatted_sources}\nSempre que citar Atenção:\n1) Inclua o nome dos documentos e número da página utilizados na resposta com o formato ('nomedodocumento', 'numero_pg')\n\nAnonimize todas as referências a nomes de pessoas.\nResponda da melhor maneira possível a seguinte \n\npergunta:{human_input}"
        return HumanMessage(content=text, additional_kwargs=self.additional_kwargs)

    def format_sources(self, data):
        formatted_text = ""
        
        for document in data['data']['Get']['Document']:
            content = document['content']
            file_name = document['fileName']
            page_or_chunk = document['pageOrChunk']
            
            formatted_text += f"### Fonte: {file_name}, Página: {page_or_chunk} ###\n"
            formatted_text += f"{content}\n"
            formatted_text += f"{'='*50}\n"
            
        return formatted_text
    
    def _prompt_type(self):
        return "dynamic-library"


#Prompts
from .prompts import SYS_PROMPT
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#Define o LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Você é a Assistente Digital da Talk, uma espécie de oraculo digital que tem acesso a todos os documentos já produzidos pela empresa. A Talk é uma empresa de pesquisa com uma metodologia bastante focada em pesquisas qualitativas, buscando identificar e encontrar usuários chaves no tema pesquisado e fazendo anáise em profundidade. Para cada pergunta do usuário, você receberá até 3 respostas do banco de dados para formular suas considerações. Traga insights e provocações relevantes sempre após uma análise."), # The persistent system prompt
    MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
    DynamicLibraryPromptTemplate.from_template("{human_input}"), # Where the human input will injected
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)