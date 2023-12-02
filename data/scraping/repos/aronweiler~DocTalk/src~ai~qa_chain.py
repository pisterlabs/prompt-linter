import os
from utilities.callback_handlers import DebugCallbackHandler

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationTokenBufferMemory

from shared.selector import get_embedding, get_chat_model, get_llm
from documents.vector_database import get_database

from ai.configurations.qa_chain_configuration import QAChainConfiguration
from ai.abstract_ai import AbstractAI
from ai.ai_result import AIResult


class QAChainAI(AbstractAI):
    def configure(self, json_args) -> None:
        self.configuration = QAChainConfiguration(json_args)
        embeddings = get_embedding(self.configuration.run_locally)
        db = get_database(embeddings, self.configuration.database_name)

        if self.configuration.chat_model:
            llm = get_chat_model(
                local=self.configuration.run_locally,
                ai_temp=float(self.configuration.ai_temp),
                max_tokens=int(self.configuration.max_tokens),
            )
        else:
            llm = get_llm(
                local=self.configuration.run_locally,
                local_model_path=self.configuration.model,
                ai_temp=float(self.configuration.ai_temp),
                max_tokens=int(self.configuration.max_tokens),
            )

        memory = (
            self._get_memory(llm, self.configuration.max_tokens)
            if self.configuration.use_memory
            else None
        )

        self.qa_chain = self._get_chain(
            llm,
            memory,
            db,
            self.configuration.top_k,
            self.configuration.chain_type,
            self.configuration.search_type,
            self.configuration.search_distance,
            self.configuration.verbose,
        )

    def _get_chain(
        self, llm, memory, db, top_k, chain_type, search_type, search_distance, verbose
    ):
        search_kwargs = {
            "search_distance": search_distance,
            "k": top_k,
            "search_type": search_type,
        }

        qa = ConversationalRetrievalChain.from_llm(
            llm,
            db.as_retriever(search_kwargs=search_kwargs),
            chain_type=chain_type,
            memory=memory,
            verbose=verbose,
            return_source_documents=True,
            callbacks=[DebugCallbackHandler()],
        )

        return qa

    def _get_memory(self, llm, max_tokens):
        memory = ConversationTokenBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
        )

        return memory

    def query(self, input):
        # If there is no memory, we have to fake it for the prompt.
        # Langchain should be better about this and automatically handle it
        if self.configuration.use_memory:
            result = self.qa_chain({"question": input})
        else:
            result = self.qa_chain({"question": input, "chat_history": []})

        source_docs = [
            {
                "document": os.path.basename(d.metadata["source"]).split(".")[0],
                "page": d.metadata["page"],
            }
            if "page" in d.metadata
            else os.path.basename(d.metadata["source"]).split(".")[0]
            for d in result["source_documents"]
        ]

        ai_results = AIResult(result, result["answer"], source_docs)

        return ai_results

