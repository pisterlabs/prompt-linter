from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

from tt_bot.meta import QAResponse
from tt_bot.logger import get_logger
from tt_bot.cache import async_cache
from tt_bot.utils.yaml_data import load_yaml


logger = get_logger(__name__)


class LgChainQA:
    def __init__(self, conf_path: str = "/resources/conf/lgchain-qa.yaml"):
        self.conf = load_yaml(conf_path)
        llm = ChatOpenAI(
            model_name=self.conf["model-name"],
            temperature=self.conf["temperature"],
            max_tokens=self.conf["max-tokens"],
        )

        self.chain = load_qa_chain(llm, chain_type=self.conf["chain-type"])

    @async_cache
    async def async_generate(
        self,
        text_chunks: list[str],
        question: str,
    ) -> QAResponse:
        documents = [Document(page_content=tc) for tc in text_chunks]
        with get_openai_callback() as callback:
            answer = await self.chain.arun(
                input_documents=documents,
                question=question,
            )

            logger.info(callback)
            qa_response = QAResponse(answer=answer)

            return qa_response
