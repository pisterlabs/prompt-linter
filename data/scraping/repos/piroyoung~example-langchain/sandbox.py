from logging import basicConfig
from logging import getLogger
from typing import List

from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import BaseMessage
from langchain.schema import BaseOutputParser
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage

basicConfig(level="INFO")
logger = getLogger(__name__)

if __name__ == "__main__":
    model: ChatOpenAI = ChatOpenAI(
        model_name="gpt-35-turbo",
        model_kwargs={"deployment_id": "gpt-35-turbo-0301"}
    )

    # 単純な問い合わせ
    answer: str = model.predict("何か面白いことを行ってください")
    logger.info(f"単純な問い合わせ: {answer}\n\n")

    # システムメッセージを設定する
    messages: List[BaseMessage] = [
        SystemMessage(content="関西弁で話してください"),
        HumanMessage(content="大阪の美味しい食べ物を教えてください"),
    ]

    answer: BaseMessage = model.predict_messages(messages)
    logger.info(f"システムメッセージを設定: {answer.content}\n\n")

    # PromptTemplateを使用する
    system_message_prompt: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(
        "{language} で話してください")
    human_message_prompt: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
        "{place} の美味しい食べ物を教えてください")
    prompt: ChatPromptTemplate = ChatPromptTemplate(messages=[system_message_prompt, human_message_prompt])

    answer: BaseMessage = model.predict_messages(prompt.format_messages(language="関西弁", place="大阪"))
    logger.info(f"PromptTemplateを使用する: {answer.content}\n\n")


    # OutputParserを使用する
    class CsvToListParser(BaseOutputParser):
        def parse(self, text: str) -> List[str]:
            return text.strip().split(",")


    system_message_prompt: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(
        """
        あなたはカンマ区切りのリストを生成するアシスタントです。
        ユーザはカテゴリを指定するので、そのカテゴリに属するものを {number} 個 リストアップしてください。
        あなたはカンマ区切りのリストのみを出力し、それ以外は決して何も出力しないでください。
        区切り文字には,を使用してください。
        """
    )
    human_message_prompt: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
        "{category}"
    )
    prompt: ChatPromptTemplate = ChatPromptTemplate(messages=[system_message_prompt, human_message_prompt])

    answer: BaseMessage = model.predict_messages(
        prompt.format_messages(category="果物", number="5")
    )

    parsed: List[str] = CsvToListParser().parse(answer.content)
    logger.info(f"OutputParserを使用する: {parsed}\n\n")

    # LLMChainを使用する
    chain: LLMChain = LLMChain(
        llm=model,
        prompt=prompt,
        output_parser=CsvToListParser()
    )
    result: List[str] = chain.run(category="果物", number="5")
    logger.info(f"LLMChainを使用する: {result}\n\n")

    # Memoryで会話を継続する
    memory: ConversationBufferMemory = ConversationBufferMemory()
    chain: ConversationChain = ConversationChain(
        llm=model,
        memory=memory
    )

    a1: str = chain.predict(input="こんにちわ")
    logger.info(f"Memoryで会話を継続する: {a1}\n\n")
    a2: str = chain.predict(input="あなたの名前は？")
    logger.info(f"Memoryで会話を継続する: {a2}\n\n")
