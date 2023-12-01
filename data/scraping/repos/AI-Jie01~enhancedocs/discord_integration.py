import os
import discord
import main
import utils
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain


class DiscordClient(discord.Client):
    async def on_ready(self):
        print(f'[Discord]: Logged in as {self.user} (ID: {self.user.id})')

    async def on_message(self, message):
        if message.author.id == self.user.id:
            return

        mention = f'<@{self.user.id}>'
        if message.content.startswith(mention):
            if utils.is_db_empty(main.config):
                await message.reply("No data found. Contact your Discord administrator. "
                                    "If you are the administrator; ingest data using "
                                    "https://github.com/enhancedocs/cli or the API directly", mention_author=True)
                return
            store = utils.get_vector_store(main.config)
            prompt = main.config.prompt
            question = message.content.split(f'{mention} ', 1)[1]
            if isinstance(message.channel, discord.Thread):
                chat_history = []
                async for msg in message.channel.history(oldest_first=True):
                    if msg.author.id == self.user.id:
                        chat_history.append(f"AI:{msg.content}")
                    else:
                        chat_history.append(f"Human:{msg.content}")
                async with message.channel.typing():
                    question_generator = LLMChain(llm=main.llm, prompt=CONDENSE_QUESTION_PROMPT)
                    doc_chain = load_qa_with_sources_chain(main.llm, chain_type="stuff", prompt=prompt)
                    chain = ConversationalRetrievalChain(
                        combine_docs_chain=doc_chain,
                        retriever=store.as_retriever(),
                        question_generator=question_generator,
                        get_chat_history=utils.get_chat_history,
                        return_source_documents=True
                    )
                    result = chain(
                        {"question": question, "project_name": main.config.project_name, "chat_history": chat_history},
                        return_only_outputs=True
                    )
                    await message.reply(content=result.get("answer"), mention_author=True)
                    return
            thread = await message.create_thread(name=question)
            async with thread.typing():
                doc_chain = load_qa_with_sources_chain(main.llm, chain_type="stuff", prompt=prompt)
                chain = RetrievalQAWithSourcesChain(
                    combine_documents_chain=doc_chain,
                    retriever=store.as_retriever(),
                    return_source_documents=True
                )
                result = chain(
                    {"question": question, "project_name": main.config.project_name},
                    return_only_outputs=True
                )
                await thread.send(content=result.get("answer"), mention_author=True)


async def start():
    intents = discord.Intents.default()
    intents.message_content = True

    client = DiscordClient(intents=intents)

    discord_token = os.environ.get("DISCORD_TOKEN")
    if discord_token is None:
        print("Discord Integration is enabled but no DISCORD_TOKEN has been provided, shutting down "
              "integration")
        return

    try:
        await client.start(discord_token)
    except KeyboardInterrupt:
        await client.close()
    except discord.errors.LoginFailure:
        print("Discord Integration is enabled but invalid DISCORD_TOKEN was provided, shutting down "
              "integration...")
