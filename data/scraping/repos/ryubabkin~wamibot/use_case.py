from langchain.chat_models import ChatOpenAI
from base.session import *

from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

import settings as sts
from llm.prompts.system import prepare_system_instruction
from utils.db_methods import load_data_from_db

class UseCase(object):
    id: str
    session: Session = Session()
    user_text: str
    instruction: str
    temperature: float = 0.5
    model = None

    def __init__(
            self,
            user_text: str,
            model: ChatOpenAI
    ):
        self.user_text = user_text
        self.model = model

    def execute(self) -> str:
        system_instruction = prepare_system_instruction()
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("system", self.instruction)
        ])
        prompt = prompt_template.format(question=self.user_text)
        answer = self.model.predict(prompt)
        return answer

    def execute_with_context(self, document_id: str = sts.COLLECTION_NAME) -> str:
        vectorstore = load_data_from_db(
            db_directory=sts.DB_FOLDER,
            collection_name=document_id
        )
        system_instruction = prepare_system_instruction()
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("system", self.instruction)
        ])
        chain_type_kwargs = {
            "prompt": prompt_template
        }
        chain = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs=chain_type_kwargs
        )
        answer = chain.run(self.user_text)
        return answer

    def run(self):
        pass


