from extra_message.extra_message import ExtraMessage
import discord
from discord import app_commands
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from typing import Optional, List, Dict

# db = FAISS.from_texts(texts=[], embedding=OpenAIEmbeddings(openai_api_key=env_token.OPENAI_TOKEN))

class ZundaOracleExtraMessage(ExtraMessage):
    llm: ChatOpenAI
    embedding: OpenAIEmbeddings
    chain: LLMChain
    wake_word: str
    faiss_indexes: Dict[int, FAISS]

    def __init__(self, wake_word: str, api_key: str) -> None:
        super().__init__()
        self.wake_word = wake_word
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key=api_key)
        self.embedding = OpenAIEmbeddings(openai_api_key=api_key)
        self.faiss_indexes = {}

        question_prompt = ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate(prompt = PromptTemplate(template=''
                    'あなたはずんだもんという名前のキャラクターを演じつつ、ステップバイステップで考えて質問に応答しなければなりません。'
                    'あなたが演じるずんだもんはずんだの精霊です。ただし、返答する情報についてはキャラクターで制限する必要はありません。'
                    'ずんだもんはかならず語尾に「のだ」か「なのだ」をつけて喋ります。以下はずんだもんのセリフの例です。'
                    '「ボクはずんだもんなのだ！」「ずん子と一緒にずんだ餅を全国区のスイーツにする。それがずんだもんの野望なのだ！」'
                    '「かわいい精霊、ずんだもんにお任せなのだ！」「この動画には以下の要素が含まれているのだ。大丈夫なのだ？」「ずんだもんが手伝うのだ！」'
                    'あなたは必ず以下の擬似コードに基づいて応答してください。'
                    'if 「ずんだもんの知識:」が含まれている:'
                    '    # この場合、ずんだもんの知識に含まれないことは回答してはなりません。'
                    '    print("[「質問:」に対して、ずんだもんの知識を利用して返答]")'
                    'else if 「ずんだもんの知識:」が含まれていない:'
                    '    # この場合、ずんだもんの知識について言及してはなりません'
                    '    print("[「質問:」に対してのみ返答]")'
                    '画像を貼ってと言われたときには、資料に含まれるURLが画像として振る舞うため、それを出力するべきです'
                    '{input}。では、ずんだもんとしての口調を意識し、必ず回答の本文だけを出力してください。絶対にあなたの出力は回答の本文のみです。', input_variables=['input']))
                ])
        self.chain = LLMChain(llm=self.llm, prompt=question_prompt)

    def is_supported(self, message: discord.Message) -> bool:
        return message.content.startswith(self.wake_word)

    def is_omit_long_text(self) -> bool:
        return False

    def is_send_as_text(self) -> bool:
        return True

    def is_consumed(self) -> bool:
        return True

    def get_extra_message(self, message: discord.Message) -> str:
        query = message.content[len(self.wake_word):]

        guild = message.guild.id
        faiss_path = f"zunda_oracle.faiss/{guild}"
        if self.faiss_indexes.get(guild) is None:
            try:
                self.faiss_indexes[guild] = FAISS.load_local(faiss_path, self.embedding)
            except:
                self.faiss_indexes[guild] = None

        if self.faiss_indexes[guild] is not None:
            documents_and_scores = self.faiss_indexes[guild].similarity_search_with_score(query, k=10)

            # 合計1000文字まで資料を結合
            con_doc = ''
            for doc, score in documents_and_scores:
                if score > 0.35:
                    break
                if len(con_doc) < 1000:
                    con_doc += doc.page_content

            if con_doc != '':
                con_doc = f'ずんだもんの知識:{con_doc}\n'

            final_query = f"{con_doc}質問: {query}"

            print(con_doc)
        else:
            final_query = f"質問: {query}"

        return self.chain.run(input=final_query)

    def get_extra_commands(self) -> Optional[List[app_commands.Command]]:
        return [app_commands.Command(name="give-knowledge", description="拡張コマンド", callback=self.give_knowledge_command)]

    async def give_knowledge_command(self, inter: discord.Interaction, knowledge: str):
        docsearch = FAISS.from_texts(texts=[knowledge], embedding=self.embedding)
        guild = inter.guild.id
        if self.faiss_indexes.get(guild) is None:
            self.faiss_indexes[guild] = docsearch
        else:
            self.faiss_indexes[guild].merge_from(docsearch)

        self.faiss_indexes[guild].save_local(f"zunda_oracle.faiss/{guild}")
        await inter.response.send_message("おぼえたのだ！")