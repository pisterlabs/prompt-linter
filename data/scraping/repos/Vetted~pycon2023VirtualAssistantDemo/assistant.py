from src.clients.qdrant import QClient
from src.clients.openai import OpenAIClient
from src.constants import DEMO_COLLECTION_NAME, QNA_PROMPT
from src.logger import logger


class Assistant:
    vector_db_client = QClient()
    llm_client = OpenAIClient()

    def get_answer(self, question: str) -> str:
        """Answers to user provided question

        Args:
            question (str): Question to answer

        Returns:
            str: Answer to the question
        """
        logger.debug(f"Fetching embeddings for the question: {question}")
        question_embedding = self.llm_client.create_embedding(question)
        relevant_contexts = self.vector_db_client.query(
            collection=DEMO_COLLECTION_NAME,
            query_embedding=question_embedding,
            top_n=5,
        )
        logger.debug("Relevant context to the question")
        for idx, relevant_context in enumerate(relevant_contexts):
            logger.debug(f"{idx}: {relevant_context[2]['content'][:20]}...")  # type: ignore
        context_map = {}  # name: context map
        for relevant_context in relevant_contexts:
            data = relevant_context[2]
            name = data["name"]
            content = data["content"]
            if name not in context_map:
                context_map[name] = []

            context_map[name].append(content)

        # Building context text
        context_items = []
        for name, item in context_map.items():
            context_items.append("\n\n")
            context_items.append(f"**{name}**")
            for line in item:
                context_items.append(f"- {line}")

        prompt = QNA_PROMPT.format(question=question, context="\n".join(context_items))
        msgs = [{"role": "user", "content": prompt}]
        logger.debug(f"LLM Prompt: {prompt}")
        answer = self.llm_client.create_chat_completion(messages=msgs)
        return answer
