"""
This file contains a cog for AI psychotherapy assistant commands.
"""

import os
import json
import time
import asyncio
import openai
import discord
from discord.ui import View, Button
from discord.ext import commands
from asgiref.sync import sync_to_async
from assets.utils.chat import Conversation, generate_conversation, num_tokens_from_messages
import assets.settings.setting as setting

logger = setting.logging.getLogger("psy")

openai.api_key = os.getenv("OPENAI_API_KEY")


class PsyGPT(commands.Cog):
    """Cog for PsyGPT commands.
    """

    def __init__(self, bot):
        self.bot = bot

        self.database_path = "assets/database/psygpt_database.json"
        self.questions_path = "assets/database/questions.json"

        self.load_database()
        self.load_questions()

        self.questionnaire_threads = {}
        self.chatting_threads = {}

    def load_questions(self):
        if not os.path.exists(self.questions_path):
            logger.error(
                f"Questions file {self.questions_path} does not exist! Make sure you have the file in the correct path.")
            raise FileNotFoundError(
                f"Questions file {self.questions_path} does not exist! Make sure you have the file in the correct path.")
        self.questions = json.load(
            open(self.questions_path, "r", encoding="utf-8"))

    def load_database(self):
        if not os.path.exists(self.database_path):
            logger.info(
                f"Database file {self.database_path} does not exist. Creating one...")
            with open(self.database_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        self.database = json.load(
            open(self.database_path, "r", encoding="utf-8"))

    def write_database(self):
        json.dump(self.database, open(
            self.database_path, "w", encoding="utf-8"), indent=4)

    @commands.Cog.listener()
    async def on_message(self, ctx):
        if ctx.author == self.bot.user:
            # message author is not the bot itself
            return
        if not ctx.guild:
            # message is from a dm
            return
        if ctx.author.id in self.questionnaire_threads \
                and ctx.channel.id == self.questionnaire_threads[ctx.author.id]["thread_id"]:
            # Questionnaire thread

            # Delete any message inside this thread that is not belong to the bot or the user.
            async for msg in ctx.channel.history(limit=None, oldest_first=True):
                if msg.author.id != ctx.author.id and msg.author.id != self.bot.user.id:
                    await msg.delete()

            user_discriminator = ctx.author.name + "#" + ctx.author.discriminator
            if self.questionnaire_threads[ctx.author.id]["counter"] == len(self.questions):
                # Make a list of all user messages in this thread in order.
                user_messages = []
                async for msg in ctx.channel.history(limit=None, oldest_first=True):
                    if msg.author.id == ctx.author.id:
                        user_messages.append(msg)

                self.database[user_discriminator] = {
                    "id": ctx.author.id,
                    "questions": self.questions,
                    "answers": [msg.content for msg in user_messages]
                }
                self.write_database()
                # Delete this user in chatting_thread
                del self.questionnaire_threads[ctx.author.id]

                logger.info(f"Saved user {user_discriminator}'s data.")

                # TODO: Analyze the qa and make a report here.
                # Pseudo code:
                report = await personality_analyze(
                    self.database[user_discriminator]["questions"],
                    self.database[user_discriminator]["answers"],
                    debug=self.bot.debug
                )
                self.database[user_discriminator]["report"] = report

                # TODO: Analyze the qa and generate a chat system prompt.
                self.database[user_discriminator]["chat_system_message"] = "你是一位熱於助人的AI助理，你的名字叫做奶辰，你是一位軟體工程師，若人類詢問你和軟體技術相關的問題，請你將人類的需求拆成一個個小問題，並且將這些問題的答案組合起來成最佳解答，回答人類的問題，並且告訴人類為何你的答案是最佳解答。確保你的答案是正確的，並且不會對人類造成傷害。若人類與你道別，請一律回答“掰掰”。若遇到專有名詞可以使用原文並以繁體中文解釋輔助，其餘請全程使用繁體中文回答。"

                self.write_database()

                await ctx.channel.send("分析已完成！將於5秒後自動關閉此討論串。")
                await asyncio.sleep(5)
                await self.close_thread(ctx.channel.id)
            else:
                await ctx.channel.send(self.questions[self.questionnaire_threads[ctx.author.id]["counter"]])
                self.questionnaire_threads[ctx.author.id]["counter"] += 1

        elif ctx.author.id in self.chatting_threads \
                and ctx.channel.id == self.chatting_threads[ctx.author.id]["thread_id"]:
            # Chatting thread

            conv = self.chatting_threads[ctx.author.id]["conversation"]
            prompt = conv.prepare_prompt(ctx.content)

            if self.bot.debug:
                logger.debug(f"\n\n{conv}\n\n")
                logger.debug(f"Tokens: {num_tokens_from_messages(prompt)}")

            if num_tokens_from_messages(prompt) > 3500:
                await ctx.reply("對話過長，請重新開始對話。")
                del self.chatting_threads[ctx.author.id]
                await self.close_thread(ctx.channel.id)
                return

            try:
                async with ctx.channel.typing():
                    if self.bot.debug:
                        if ctx.content == "掰掰":
                            # Debuging chat exit function
                            full_reply_content = "掰掰"
                        else:
                            # Debuging reply function
                            full_reply_content = "這是一個測試回應。為了避免過度使用 OpenAI API，這個回應是從本地讀取的。"
                    else:
                        start_time = time.time()
                        response = await sync_to_async(openai.ChatCompletion.create)(
                            model="gpt-3.5-turbo",
                            messages=prompt,
                            stream=True
                        )
                        collected_messages = []
                        message = None
                        for chunk in response:
                            chunk_time = time.time() - start_time  # calculate the time delay of the chunk
                            # extract the message
                            chunk_message = chunk['choices'][0]['delta']
                            collected_messages.append(
                                chunk_message)  # save the message
                            if chunk_time > 3.0:
                                full_reply_content = ''.join(
                                    [m.get('content', '') for m in collected_messages])
                                if message:
                                    await message.edit(content=full_reply_content)
                                else:
                                    message = await ctx.channel.send(full_reply_content)
                                start_time = time.time()

                        full_reply_content = ''.join(
                            [m.get('content', '') for m in collected_messages])
                        if message:
                            await message.edit(content=full_reply_content)
                        else:
                            message = await ctx.channel.send(full_reply_content)

            except Exception as e:
                logger.error(f"Failed to generate conversation: {e}")
                await ctx.reply(f"生成對話時發生錯誤：{e}")

            if full_reply_content == "":
                await ctx.reply("沒有生成任何回應。")
                return

            conv.append_response(full_reply_content)

            # If the bot reply with "掰掰", end the conversation
            if "掰掰" in full_reply_content:
                logger.debug("Quitting Chat...")
                await asyncio.sleep(3)
                del self.chatting_threads[ctx.author.id]
                await self.close_thread(ctx.channel.id)

    @commands.command(name="update_psygpt_api_key")
    @commands.has_permissions(administrator=True)
    async def _update_api_key(self, ctx, key):
        if key is None:
            return
        await ctx.defer()
        openai.api_key = key
        await ctx.send(f"Updated OpenAI API key")

    @commands.hybrid_command(name="analyze", description="Analyze your personality.")
    async def _analyze(self, ctx):
        await ctx.defer()

        user_id = ctx.author.id

        ids_in_database = [self.database[user]["id"] for user in self.database]
        if user_id in ids_in_database:
            # Send a message contains yes button and no button ask if the user wants to overwrite the data, if user choose yes, then keep the function, if no, return.
            view = View()
            view.add_item(
                Button(label="是", style=discord.ButtonStyle.blurple, emoji="✅"))
            view.add_item(
                Button(label="否", style=discord.ButtonStyle.gray, emoji="❌"))

            message = await ctx.send("是否要覆蓋已記錄的資料？", view=view)

            def check(res):
                return res.data["component_type"] == 2 and res.user.id == ctx.author.id and res.message.id == message.id

            try:
                res = await self.bot.wait_for("interaction", timeout=10.0, check=check)
                custom_id = res.data["custom_id"]
                clicked_button = None
                for child in view.children:
                    if isinstance(child, Button) and child.custom_id == custom_id:
                        clicked_button = child
                        break
                logger.debug(f"Clicked button: {clicked_button.label}")
                if clicked_button is not None and clicked_button.label == "是":
                    await message.edit(content="資料會在分析完成後覆蓋。", view=None)
                else:
                    await message.edit(content="已取消。", view=None)
                    return
            except asyncio.TimeoutError:
                await message.edit(content="請重新輸入指令。", view=None)

        # Check if the user has already started a conversation in a thread. If yes, send a message that mention the thread to the user.
        if user_id in self.questionnaire_threads:
            thread_id = self.questionnaire_threads[user_id]
            try:
                thread = await self.bot.fetch_channel(thread_id)
            except Exception as e:
                # If the thread has been deleted, remove the thread from the dictionary.
                del self.questionnaire_threads[user_id]
                await ctx.send("請重新開始一次分析。")
                return
            await ctx.send(f"你已經在 <#{thread.id}> 裡面開始了分析。")
            return

        msg = await ctx.send("已開始分析")
        thread = await ctx.channel.create_thread(
            name=f"{ctx.author.name} 的分析", message=msg, auto_archive_duration=60)
        self.questionnaire_threads[user_id] = {
            "thread_id": thread.id,
            "counter": 0
        }

        await thread.send(self.questions[0])
        self.questionnaire_threads[user_id]["counter"] += 1

    @commands.hybrid_command(name="self_chat", description="Chat with you. Yes, you.")
    async def _self_chat(self, ctx):
        await ctx.defer()
        user_id = ctx.author.id
        user_discriminator = ctx.author.name + "#" + ctx.author.discriminator

        ids_in_database = [self.database[user]["id"] for user in self.database]
        if user_id not in ids_in_database:
            await ctx.send("你還沒有進行分析！")
            return

        msg = await ctx.send("已開始分析")
        thread_name = f"{user_discriminator} 與自己的聊天室"
        thread = await ctx.channel.create_thread(
            name=thread_name,
            message=msg,
            auto_archive_duration=60,
        )

        self.load_database()
        conversation = Conversation()
        conversation.init_system_message(
            self.database[user_discriminator]["chat_system_message"])
        self.chatting_threads[ctx.author.id] = {
            "thread_id": thread.id,
            "conversation": conversation
        }

    @commands.hybrid_command(name="report", description="Get your personality report.")
    async def _report(self, ctx):
        """Return the user's personality report by dm."""
        await ctx.defer()

        user_id = ctx.author.id

        ids_in_database = [self.database[user]["id"] for user in self.database]
        if user_id not in ids_in_database:
            await ctx.send("你還沒有進行分析！")
            return

        user_discriminator = ctx.author.name + "#" + ctx.author.discriminator
        user_data = self.database[user_discriminator]["report"]

        await ctx.author.send(user_data)

        await ctx.send("已將分析結果私訊給你。")

    async def close_thread(self, id):
        """Delete the thread"""
        try:
            thread = await self.bot.fetch_channel(id)
            await thread.delete()
            return True
        except Exception as e:
            logger.error(f"Failed to close thread {id} with error {e}")
            return False


async def personality_analyze(questions: list, answers: list, debug: bool = False):
    """Analyze the user's personality by their answers.

    Args:
        questions (list): A list of questions.
        answers (list): A list of answers.

    Returns:
        string: A string contains the user's personality report.
    """

    # Example to use Conversation and generate_conversation to call OpenAI ChatGPT API.
    conv = Conversation()
    conv.init_system_message("Test system message.")

    prompt = conv.prepare_prompt(questions + answers)

    if not debug:
        completion = await generate_conversation(prompt)
    else:
        completion = "Debug message"

    return completion


async def setup(client):
    await client.add_cog(PsyGPT(client))
