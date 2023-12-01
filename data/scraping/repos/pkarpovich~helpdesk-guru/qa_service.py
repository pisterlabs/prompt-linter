from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import messages_to_dict, messages_from_dict

from gpt_context.adapters import VectorStoreAdapter
from gpt_context.exceptions import BusinessLogicException, ErrorCode, ErrorMessage
from gpt_context.services.prompts.qa_prompt import qa_prompt
from gpt_context.stores import ConversationStore

callbacks = [StreamingStdOutCallbackHandler()]


class QAService:
    def __init__(self, model_name: str, store: VectorStoreAdapter, conversation_store: ConversationStore):
        self.conversation_store = conversation_store
        self.history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(chat_memory=self.history, memory_key="chat_history",
                                               return_messages=True)
        self.store = store

        llm = ChatOpenAI(model_name=model_name, callbacks=callbacks, temperature=0)

        self.question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
        self.combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt, verbose=True)

    def ask(self, query: str, conversation_id: str, context_name: str) -> str:
        if not self.store.context_exists(context_name):
            raise BusinessLogicException(
                ErrorCode.NOT_FOUND,
                ErrorMessage.CONTEXT_NOT_FOUND.format(context_name=context_name)
            )

        self._load_messages_if_needed(conversation_id)

        qa = ConversationalRetrievalChain(
            memory=self.memory,
            retriever=self.store.get_retriever(context_name),
            question_generator=self.question_generator,
            combine_docs_chain=self.combine_docs_chain,
            verbose=True,
        )

        res = qa({"question": query})
        self.conversation_store.save_or_update(
            conversation_id,
            messages_to_dict(self.memory.chat_memory.messages)
        )

        return res["answer"]

    def clear_history(self, conversation_id: str) -> None:
        self.conversation_store.delete_by_id(conversation_id)
        self.memory.clear()

    def _load_messages_if_needed(self, conversation_id: str):
        self.memory.clear()

        conversation = self.conversation_store.get_by_id(conversation_id)
        if conversation is None:
            return

        self.memory.chat_memory.messages = messages_from_dict(conversation["messages"])
