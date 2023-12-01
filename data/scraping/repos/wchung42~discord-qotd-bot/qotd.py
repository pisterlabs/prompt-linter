import discord
from discord import app_commands
from discord.ext import commands, tasks
from discord.ui import View
from datetime import datetime, time
from dateutil.tz import gettz
from typing import Optional
import asyncpg
import os
import postgres
import utils
import openai
import random
import asyncio

# ----------------------------------
# QOTD utility functions
# ----------------------------------

# Get question from OpenAI
def fetch_questions(asked_questions: list[str] = None, *args, **kwargs) -> list[str]:
    '''
    Fetch 45 new questions from from OpenAI
    '''
    openai.api_key = (str)(os.getenv('OPENAI_API_KEY'))
    success: bool = False
    delay: float = 1
    exponential_base: float = 2
    jitter: bool = True
    num_retries: int = 0
    max_retries: int = 5
    
    prompt: str = 'Give me a newline delimited list of 45 unique conversation starter questions'
    if asked_questions:
        prompt += f' that is not already in the following list: {asked_questions}' 
    
    while not success:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant that asks thought-provoking and conversation starter questions.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.67,
                max_tokens=2250,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            # print(response)
        except openai.APIError as e:
            num_retries += 1
            if num_retries > max_retries:
                return None
            delay *= exponential_base * (1 + jitter * random.random())
            asyncio.sleep(delay)
        except openai.InvalidRequestError as e:
            continue
        except Exception as e:
            continue
        else:
            success = True
            questions_content: str = response.choices[0]['message']['content'].strip('\"')
            invalid_responses: list = [
                "I'm sorry, I cannot generate inappropriate or offensive content.", "AI language model",]
            for invalid in invalid_responses:
                if invalid in questions_content:
                    success = False
                    break
            if success:
                # Format questions string as a list
                questions_list: list[str] = []
                for question in questions_content.split('\n'):
                    try:
                        questions_list.append(question.split('.', 1)[1].strip())
                    except Exception as e:
                        pass
                return questions_list
    return None


async def update_questions_list(bot: commands.Bot, guild_id: int, *args, **kwargs) -> bool:
    '''
    Update database with new unasked_questions
    and reset asked_questions for the given guild
    '''
    # Fetch new questions
    loop = asyncio.get_running_loop()
    new_questions: list[str] = await loop.run_in_executor(None, fetch_questions)
    if not new_questions:
        return False
    
    # Update database
    try:
        query: str = '''
            UPDATE guilds
            SET unasked_questions = $1, asked_questions = $2
            WHERE guild_id = $3;
        '''
        await bot.db.execute(query, new_questions, [], guild_id)
    except asyncpg.PostgresError as e:
        await postgres.send_postgres_error_embed(bot, query, e)
        return False
    
    return True


async def get_question(bot: commands.Bot, guild_id: int, *args, **kwargs) -> str:
    '''
    Returns the last question in unasked_questions
    and moves it from unasked_questions to asked_questions.
    '''
    try:
        query: str = '''
            SELECT unasked_questions, asked_questions
            FROM guilds
            WHERE guild_id = $1
        '''
        result = await bot.db.fetchrow(query, guild_id)
    except asyncpg.PostgresError as e:
        await postgres.send_postgres_error_embed(bot, query, e)
    
    if result:
        unasked_questions: list[str] = result.get('unasked_questions')
        asked_questions: list[str] = result.get('asked_questions')
        
        # Generate new questions if there are no more questions in unasked
        if not unasked_questions:
            loop = asyncio.get_running_loop()
            unasked_questions = await loop.run_in_executor(None, fetch_questions, asked_questions)

        # Early return if error generating questions
        if not unasked_questions:
            # Replace print with discord logging
            print('[ERROR]: Could not generate new questions')
            return None
        
        # Update asked_questions for guild
        qotd_question: str = unasked_questions[-1]
        updated_unasked_questions: list[str] = unasked_questions[0:len(unasked_questions) - 1]
        asked_questions.append(qotd_question)
        updated_asked_questions: list[str] = asked_questions
        try:
            query: str = '''
                UPDATE guilds
                SET unasked_questions = $1, asked_questions = $2
                WHERE guild_id = $3;
            '''
            await bot.db.execute(query, updated_unasked_questions, updated_asked_questions, guild_id)
        except asyncpg.PostgresError as e:
            await postgres.send_postgres_error_embed(bot, query, e)
        else:
            return qotd_question
    
    print('[ERROR]: Failed to fetch from postgres database')
    return None


# ----------------------------------
# QOTD classes
# ----------------------------------
class PendingQOTDView(View):
    def __init__(self, bot: commands.Bot):
        super().__init__(timeout=86400)
        self.bot = bot
    

    @discord.ui.button(label='Approve', style=discord.ButtonStyle.green, emoji='👍')
    async def approve_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Get qotd channel
        try:
            query: str = '''
                SELECT qotd_channel_id
                FROM guilds
                WHERE guild_id = $1
            '''
            response = await self.bot.db.fetchrow(query, interaction.guild_id)
        except asyncpg.PostgresError as e:
            await postgres.send_postgres_error_embed(bot=self.bot, query=query, error_msg=e)
        
        if response:
            qotd_channel: discord.abc.GuildChannel = self.bot.get_channel(response.get('qotd_channel_id'))

            # Edit original embed to show "Approved" under status
            if interaction.message.embeds[0]: # Edit field if embed exists
                updated_pending_qotd_embed: discord.Embed = interaction.message.embeds[0].set_field_at( 
                    index=1,
                    name='Status',
                    value='Approved ✅',
                    inline=False
                )
                await interaction.response.edit_message(embed=updated_pending_qotd_embed, view=None) # Disable buttons from pending message

            # Create qotd embed
            qotd_embed: discord.Embed = discord.Embed(
                title=f'<:question:956191743743762453><:grey_question:956191743743762453>'
                    f'Question of The Day<:grey_question:956191743743762453><:question:956191743743762453>',
                description=f'{interaction.message.embeds[0].fields[0].value}', 
                color=0xFC94AF,
                timestamp=datetime.now()
            )
            qotd_message: discord.Message = await qotd_channel.send(embed=qotd_embed)
            

    @discord.ui.button(label='Reroll', style=discord.ButtonStyle.red, emoji='🔁')
    async def callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        question: str = None
        while True:
            question: str = await get_question(self.bot, interaction.guild_id)
            if question:
                break

        old_embed: discord.Embed = interaction.message.embeds[0]
        if old_embed:
            new_embed: discord.Embed = old_embed.set_field_at(
                index=0,
                name='Question',
                value=question,
                inline=False
            ) 
            await interaction.response.edit_message(embed=new_embed)
    

    def on_timeout(self):
        self.clear_items() # Clear view on timeout

            
# ---------------------------
# QOTD cog implementation
# ---------------------------
class Qotd(commands.Cog):
    '''Question of The Day Cog'''
    OWNER_GUILD_ID = os.getenv('OWNER_GUILD_ID')

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.bot.tree.on_error = self.cog_app_command_error # Add error handler to bot tree
        

    async def cog_load(self) -> None:
        self.qotd_send_question.start() # Start task
        print('* QOTD module READY')


    async def cog_unload(self) -> None:
        '''Gracefully stops all tasks from running'''
        self.qotd_send_question.stop()


    async def cog_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        '''Error handler for QOTD module'''
        to_send: str = '**[ERROR]:** '
        if isinstance(error, app_commands.MissingPermissions):
            to_send += f'You are missing `{error.missing_permissions}` permission(s) to use this command in this channel.'
        elif isinstance(error, app_commands.BotMissingPermissions):
            to_send += f'I am missing `{error.missing_permissions}` permission(s) to use this command in this channel.'
        elif isinstance(error, discord.NotFound):
            to_send += f'Could not find channel.\n\n{error}'
        elif isinstance(error, discord.Forbidden):
            to_send += f'{error}'
        else:
            to_send += f'{error}'

        await interaction.response.send_message(to_send, ephemeral=True)    


    def create_pending_question_embed(self, question: str) -> discord.Embed:
        pending_question_embed: discord.Embed = discord.Embed(
            title='Pending QOTD',
            color=0xa7c7e7,
            timestamp=datetime.now()
        )
        pending_question_embed.add_field(name='Question', value=question, inline=False)
        pending_question_embed.add_field(name='Status', value='Pending', inline=False)
        return pending_question_embed
    

    # ----------------------------------
    # QOTD commands
    # ----------------------------------

    # QOTD setup command
    @app_commands.command(name='setup', description='Set up Question of the Day.')
    @app_commands.checks.has_permissions(manage_guild=True, manage_channels=True)
    @app_commands.checks.bot_has_permissions(manage_channels=True, send_messages=True)
    async def qotd_setup(
        self, 
        interaction: discord.Interaction, 
        channel: Optional[discord.TextChannel]
    ) -> None:
        '''Setup command for QOTD module.'''
        await interaction.response.defer()
        # Check if server requires QOTD setup
        try:
            query: str = '''
                SELECT qotd_channel_id, qotd_approval_channel_id
                FROM guilds 
                WHERE guild_id = $1;
            '''
            response = await self.bot.db.fetch(query, interaction.guild_id)
        except asyncpg.PostgresError as e:
            await postgres.send_postgres_error_embed(bot=self.bot, query=query, error_msg=e)
            await interaction.followup.send(f'**[ERROR]**: Something went wrong. Please try again.', ephemeral=True)
            return 

        # Set given channel as QOTD channel if given, else create QOTD channel
        qotd_channel: discord.abc.GuildChannel = interaction.guild.get_channel(response[0].get('qotd_channel_id'))
        qotd_approval_channel: discord.abc.GuildChannel = interaction.guild.get_channel(response[0].get('qotd_approval_channel_id'))
        if qotd_channel and qotd_approval_channel:
            await interaction.followup.send(f'QOTD is already set up in {qotd_channel.mention} and {qotd_approval_channel.mention}. '\
                                            f'If you want to edit channels, please use `channel`.', ephemeral=True)
            return
        else:
            overwrites = {
                interaction.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                interaction.guild.me: discord.PermissionOverwrite( # Necessary permissions for bot
                    read_messages=True,
                    send_messages=True,
                    embed_links=True,
                    read_message_history=True,
                )
            }

            # Check if channel given is a text channel
            if channel and isinstance(channel, discord.TextChannel):
                required_command_perms: list[tuple] = {
                    ('read_messages', True), ('send_messages', True), 
                    ('embed_links', True), ('read_message_history', True)
                }
                missing_perms: list = utils.perms_check(interaction.guild.get_member(self.bot.user.id), channel, required_command_perms)
                if missing_perms:
                    await interaction.followup.send(f'Bot is missing these required permissions' \
                                                            f'`{missing_perms}` in {channel.mention} for QOTD.', ephemeral=True)
                    return
                qotd_channel = channel
            # Create QOTD channel if no channel or incorrect channel given
            else:
                qotd_channel: discord.abc.GuildChannel = await interaction.guild.create_text_channel(
                    name='qotd', 
                    reason='QOTD setup', 
                    overwrites=overwrites
                )
            
            # Create approval channel
            if qotd_channel:
                qotd_approval_channel: discord.abc.GuildChannel = await interaction.guild.create_text_channel(
                    name='qotd-approval',
                    reason='QOTD setup: QOTD approval channel',
                    overwrites=overwrites
                )

            if qotd_channel and qotd_approval_channel:
                success_message_text: str = f'**[SUCCESS]:** Questions for approval will appear in {qotd_approval_channel.mention}' \
                                       f' and approved questions will appear in {qotd_channel.mention}'
                try:
                    query = '''
                        UPDATE guilds 
                        SET qotd_channel_id = $1, qotd_approval_channel_id = $2
                        WHERE guild_id = $3
                    '''
                    await self.bot.db.execute(query, qotd_channel.id, qotd_approval_channel.id, interaction.guild_id)
                except asyncpg.PostgresError as e:
                    await postgres.send_postgres_error_embed(self.bot, query, e)
                    return
                else:
                    message: discord.WebhookMessage = await interaction.followup.send(success_message_text + '\n**Generating new questions...**', wait=True)
                
                    updated: bool = await update_questions_list(self.bot, interaction.guild_id)
                    if updated:
                        await message.edit(content=message.content + '\n**[SUCCESS]:** New questions have been **added** for your server.')
                    else:
                        await message.edit(message=message.content + '\n**[ERROR]:** Failed to add new questions for your server. Please use `/update`.')
            else:
                await interaction.followup.send('**[ERROR]:** Failed to set up QOTD. Please try again.', ephemeral=True)


    @app_commands.command(name='channel', description='Get or edit QOTD channel')
    @app_commands.checks.has_permissions(manage_channels=True)
    @app_commands.checks.bot_has_permissions(send_messages=True)
    async def qotd_edit_channel(self, interaction: discord.Interaction, channel: Optional[discord.TextChannel]) -> None:
        """Command to update QOTD channel."""
        # Required permissions for this command
        interaction_msg: str = ''
        # If channel given, update qotd channel
        if channel:
            if isinstance(channel, discord.TextChannel):
                # Required permissions for this command
                required_command_perms = {('read_messages', True), ('send_messages', True), ('embed_links', True), ('read_message_history', True)}
                missing_perms: list = utils.perms_check(self.bot, channel, required_command_perms)

                # If bot is missing permissions, send user list of missing permissions
                if missing_perms:
                    interaction_msg += f'Bot is missing these required permissions `{missing_perms}` in {channel.mention} for QOTD.'
                else:
                    try:
                        query = '''
                            UPDATE guilds 
                            SET qotd_channel_id = $1 
                            WHERE guild_id = $2
                            '''
                        await self.bot.db.execute(query, channel.id, interaction.guild_id)
                    except asyncpg.PostgresError as e:
                        await postgres.send_postgres_error_embed(bot=self.bot, query=query, error_msg=e)
                    else:
                        interaction_msg += f'QOTD is set to {channel.mention}.'
            else:
                interaction_msg += 'That is **not** a valid text channel.'
        else:
            # Fetch QOTD channel id from database
            try:
                query = '''
                    SELECT qotd_channel
                    FROM guilds 
                    WHERE guild_id = $1;
                '''
                qotd_channel_id = await self.bot.db.fetchval(query, interaction.guild_id) # Fetch channel id
            except asyncpg.PostgresError as e:
                await postgres.send_postgres_error_embed(bot=self.bot, query=query, error_msg=e)
                interaction_msg += 'Could not set up QOTD. Please try again.'
            else:
                if qotd_channel_id:
                    qotd_channel = await self.bot.fetch_channel(qotd_channel_id) # Fetch channel 
                
                    if qotd_channel:
                        interaction_msg += f'Current QOTD channel is {qotd_channel.mention}.'
                    else:
                        interaction_msg += f'The channel that QOTD is linked to does not exist. Please remove and set up QOTD again.'
                else:
                    interaction_msg += 'QOTD is not set up in this server.'

        await interaction.response.send_message(interaction_msg)


    @app_commands.command(name='remove', description='Removes QOTD from server.')
    @app_commands.describe(confirmation='Select "Yes" to confirm removal.')
    @app_commands.choices(confirmation=[
        app_commands.Choice(name='Yes', value=1),
        app_commands.Choice(name='No', value=0)
    ])
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.checks.bot_has_permissions(manage_channels=True, send_messages=True)
    async def qotd_remove(
        self,
        interaction: discord.Interaction,
        confirmation: app_commands.Choice[int]
    ) -> None:
        '''Removes QOTD from the server.'''
        interaction_msg: str = ''

        if confirmation.value == 1:
            # Fetch QOTD channel from db
            try:
                query: str = '''
                    SELECT qotd_channel_id, qotd_approval_channel_id 
                    FROM guilds
                    WHERE guild_id = $1
                '''
                result = await self.bot.db.fetchrow(query, interaction.guild_id)
            except asyncpg.PostgresError as e:
                await postgres.send_postgres_error_embed(bot=self.bot, query=query, error_msg=e)
            
            if result:
                qotd_channel_id: int = result.get('qotd_channel_id')
                qotd_approval_channel_id: int = result.get('qotd_approval_channel_id')
                if qotd_channel_id and qotd_approval_channel_id:
                    # Remove qotd_channel_id and qotd_approval_channel_id from database
                    try:
                        query: str = '''
                            UPDATE guilds
                            SET qotd_channel_id = NULL, qotd_approval_channel_id = NULL, unasked_questions = '{}', asked_questions = '{}'
                            WHERE guild_id = $1
                        '''
                        await self.bot.db.execute(query, interaction.guild_id)
                    except asyncpg.PostgresError as e:
                        await postgres.send_postgres_error_embed(bot=self.bot, query=query, error_msg=e)
                    else:
                        interaction_msg += 'QOTD removed.'
                    
                    # Delete qotd approval channel
                    qotd_approval_channel: discord.TextChannel = interaction.guild.get_channel(qotd_approval_channel_id)
                    if qotd_approval_channel:
                       await qotd_approval_channel.delete(reason='Removed QOTD.')
                else:
                    interaction_msg += 'QOTD is not set up. Use `/setup`.'   
        else:
            interaction_msg += 'No action performed.'
        
        await interaction.response.send_message(interaction_msg)


    @app_commands.command(name='send', description='Manually sends "Question of the Day".')
    @app_commands.checks.has_permissions(administrator=True)
    async def qotd_manual_send(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        try:
            query: str = '''
                SELECT qotd_approval_channel_id
                FROM guilds
                WHERE guild_id = $1
            '''
            result = await self.bot.db.fetchrow(query, interaction.guild_id)
        except asyncpg.PostgresError as e:
            await postgres.send_postgres_error_embed(bot=self.bot, query=query, error_msg=e)
        
        if result:
            qotd_approval_channel_id: str = result.get('qotd_approval_channel_id')
            channel: discord.abc.GuildChannel = await self.bot.fetch_channel(qotd_approval_channel_id)
            if not channel:
                await interaction.followup.send('**[ERROR]:** No valid QOTD channel found. Try **/setup**.', ephemeral=True)
                return

            question: str = await get_question(self.bot, interaction.guild_id)
            if not question:
                await interaction.followup.send('**[ERROR]:** Cannot fetch question. Please try again.', ephemeral=True)
                return
            
            # Send embed
            pending_question_embed: discord.Embed = self.create_pending_question_embed(question)
            message: discord.Message = await channel.send(embed=pending_question_embed, view=PendingQOTDView(self.bot))
            if message:
                await interaction.followup.send(f'**[SUCCESS]:** Sent QOTD to {channel.mention}.', ephemeral=True)
            else:
                await interaction.followup.send(f'**[ERROR]:** Failed to send QOTD to {channel.mention}. Please try again.', ephemeral=True)
        else:
            await interaction.followup.send('No valid QOTD channel found. Try **/setup**.', ephemeral=True)
            

    @tasks.loop(time=[time(10, 0, 0, 0, tzinfo=gettz('US/Eastern'))], reconnect=True)
    async def qotd_send_question(self) -> None:
        '''Task sends QOTD @10AM EST daily.'''
        # Get all pending channels to send QOTD to
        try:
            query = '''
                SELECT qotd_approval_channel_id
                FROM guilds
                WHERE qotd_approval_channel_id IS NOT NULL
            '''
            results = await self.bot.db.fetch(query)
        except asyncpg.PostgresError as e:
            await postgres.send_postgres_error_embed(bot=self.bot, query=query, error_msg=e)
        
        if results:
            channels_to_send: list = [res.get('qotd_approval_channel_id') for res in results]
            # Get question prompt and send to channel
            for channel_id in channels_to_send:
                channel = await self.bot.fetch_channel(channel_id)
                question: str = await get_question(self.bot, channel.guild.id)
                if not question:
                    await channel.send('**[ERROR]:** Error encountered with QOTD. Please try using `/send` instead.')
                    continue   
                pending_question_embed: discord.Embed = self.create_pending_question_embed(question)
                message: discord.Message = await channel.send(embed=pending_question_embed, view=PendingQOTDView(self.bot))

    
    @qotd_send_question.error
    async def send_question_error(ctx, error):
        if isinstance(error, commands.ChannelNotFound):
            pass
    

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Qotd(bot))