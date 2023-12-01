import discord
from openai import RateLimitError, APIStatusError

from gpt import GPT, GPTModel


class RetryButton(discord.ui.View):
    """
    This class is invoked when `openai.error.RateLimitError` is thrown in `collect_and_send` method.
    It is responsible for rendering retry button and handling the retry mechanism.
    """

    @discord.ui.button(label='Retry', style=discord.ButtonStyle.primary)
    async def retry_callback(self, interaction: discord.Interaction, _: discord.ui.Button):
        """
        This function handles user press on the retry button. It deletes the original retry message,
            and re-tries to send payload.
        Args:
            interaction: The interaction object which contains the context of the chat
            _: button object, ignored
        """
        await interaction.message.delete()
        await DiscordUtil.collect_and_send(interaction.channel)


class DiscordUtil:
    """
    This class is responsible for all interactions with the Discord API.
    """

    @staticmethod
    async def safe_send(ctx: discord.abc.Messageable, text: str, view: discord.ui.View = None) -> None:
        """
        Sends a message to the given channel or user, breaking it up if necessary to avoid Discord's message length limit.
        """
        while len(text) > 2000:
            break_pos = text.rfind('\n', 0, 2000)
            await ctx.send(text[:break_pos])
            text = text[break_pos:]
        await ctx.send(text, view=view if view else None)

    @staticmethod
    async def collect_and_send(thread: discord.Thread, gpt_client: GPT) -> None:
        """
        Collects history messages from the thread, constructs a GPT payload, and sends the assistant's response.

        This function is responsible for collecting messages from the given thread, constructing a payload
        to send to the GPT model, and sending the assistant's response back to the thread.

        If a RateLimitError occurs, it renders a RetryButton for retrying the process.

        Args:
            thread (discord.Thread): The thread where messages are collected and the assistant's response is sent.
            gpt_client (GPT): The GPT client used to send the payload to the GPT model.

        Raises:
            openai.error.RateLimitError: If the rate limit is exceeded for the GPT API call.
        """
        async with thread.typing():
            try:
                assistant_response = await gpt_client.communicate(thread)
                await DiscordUtil.safe_send(thread, assistant_response)
            except RateLimitError as ex:
                # Render retry button on rate limit
                await DiscordUtil.safe_send(thread, ex.user_message, view=RetryButton())
            except Exception as ex:
                await DiscordUtil.safe_send(thread, ex.message)

    @staticmethod
    async def initiate_thread(message: discord.Message):
        """
        Initiates a thread in Discord.

        Parameters:
            message (discord.Message): The message object that triggered the function.
        """
        thread = await message.create_thread(name=f'Using model: {GPT.DEFAULT_MODEL.version}')
        await thread.send(**DiscordUtil.generate_model_options())
        await thread.send(**DiscordUtil.generate_temperature_options())
        await thread.send(**DiscordUtil.generate_top_p_value_options())

    @staticmethod
    def generate_model_options(selected_model: GPTModel = GPT.DEFAULT_MODEL):
        """
        Generates GPT model options as Select Menu.
        """
        view = discord.ui.View()
        for model in GPTModel:
            if model == selected_model:
                button = discord.ui.Button(style=discord.ButtonStyle.success, row=0, label=model.version,
                                           custom_id=f'model_{model.version}', disabled=True)
            else:
                button = discord.ui.Button(style=discord.ButtonStyle.primary, row=0, label=model.version,
                                           custom_id=f'model_{model.version}', disabled=not model.available)
            view.add_item(button)
        return {
            'content': '**Model**:',
            'view': view
        }

    @staticmethod
    def generate_temperature_options(selected_temperature: float = GPT.DEFAULT_TEMPERATURE):
        """
        Generates temperature options as Select Menu.
        """
        view = discord.ui.View()
        select_menu = discord.ui.Select(custom_id='temperature_select', placeholder='Select Temperature')
        for temperature in [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]:
            select_menu.add_option(label=f'{temperature}', value=f'{temperature}', default=(temperature == selected_temperature))
        view.add_item(select_menu)
        return {
            'content': '**Temperature** (controls the randomness of the generated responses):',
            'view': view
        }

    @staticmethod
    def generate_top_p_value_options(selected_top_p: float = GPT.DEFAULT_TOP_P):
        """
        Generates top p value options as Select Menu.
        """
        view = discord.ui.View()
        select_menu = discord.ui.Select(custom_id='top_p_select', placeholder='Select Top P Value')
        for top_p in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
            select_menu.add_option(label=f'{top_p}', value=f'{top_p}', default=(top_p == selected_top_p))
        view.add_item(select_menu)
        return {
            'content': '**Top P value** (controls the diversity and quality of the responses):',
            'view': view
        }

    @staticmethod
    def extract_set_value(select_message):
        content = select_message.content.lower()
        component = select_message.components.pop()
        if 'model' in content:
            for button in component.children:
                if button.style is discord.ButtonStyle.success:
                    return button.label
        else:
            child = component.children.pop()
            for option in child.options:
                if option.default:
                    return option.label
        return None
