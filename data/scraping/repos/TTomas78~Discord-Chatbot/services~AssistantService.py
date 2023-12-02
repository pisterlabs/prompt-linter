from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from data.profiles import SystemProfile,UserProfile
from data.template import template
class AssistantService():
    def __init__(self, pinecone):
        self.system_profile = SystemProfile().job_data
        self.user_profile = UserProfile().job_data
        self.chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
        self.pinecone = pinecone
        self.template = template
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.template)

        self.prompt_messages = [system_message_prompt]

    def send_message(self,query, k=4):
        """
        gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
        the number of tokens to analyze.
        """
        db = self.pinecone.get_docsearch()
        docs = db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])
        # Template to use for the system message prompt


        self.prompt_messages.append(HumanMessagePromptTemplate.from_template(query))

        chat_prompt = ChatPromptTemplate.from_messages(self.prompt_messages)

        chain = LLMChain(llm=self.chat, prompt=chat_prompt, verbose=True)

        response = chain.run(docs=docs_page_content, system_profile=self.system_profile, user_profile=self.user_profile)

        self.prompt_messages.append(SystemMessagePromptTemplate.from_template(response))

        response = response.replace("\n", "").replace("¿","").replace("¡","").replace("é","e").replace("á","a").replace("í","i").replace("ó","o").replace("ú","u")
        #add the AI response to the prompt messages
        return response
