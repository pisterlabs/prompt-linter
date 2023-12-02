import os
import openai
import discord
from discord import app_commands
import random
from datetime import datetime, timedelta
from core.classes import Cog_Extension

context_store = {}


class Slash(Cog_Extension):
    @app_commands.command(name="register", description="Зарегистрируйтесь в SkillCraft Studio")
    async def register_user(self, interaction: discord.Interaction, api_key: str, prompt_name: str = None):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/register` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id).split("#")[0]
        users_file_path = 'users.txt'
        user_folder_path = f'users/{user_id}'
        openai_folder_path = f'{user_folder_path}/openai'
        skills_folder_path = f'{user_folder_path}/skills'
        prompt_file_path = f'{user_folder_path}/{prompt_name}.txt'
        key_file_path = f'{openai_folder_path}/key.txt'
        temporary_prompt_file_path = f'{openai_folder_path}/temporary_prompt.txt'

        with open(users_file_path, 'r') as f:
            if f'{user_id}#' in f.read():
                await interaction.response.send_message("Ошибка: Вы уже зарегистрированы.", ephemeral=True)
                return

        if not api_key.startswith("sk-") or len(api_key) > 75:
            await interaction.response.send_message("Ошибка: Некорректный токен. Вы также можете купить токен, воспользовавшись командой `/buy-key`",
                ephemeral=True)
            return

        registration_date = datetime.now().strftime("%d %B %Y г.")
        user_data = f'{user_id}#{registration_date}\n'

        with open(users_file_path, 'a') as f:
            f.write(user_data)

        os.makedirs(user_folder_path, exist_ok=True)
        os.makedirs(openai_folder_path, exist_ok=True)
        os.makedirs(skills_folder_path, exist_ok=True)
        open(key_file_path, 'w').write(api_key)
        open(temporary_prompt_file_path, 'w').close()
        open(prompt_file_path, 'w').close()

        await interaction.response.send_message("Вы успешно зарегистрировались. Рекомендуем ознакомится с [документацией](https://docs.kazetech.ru/skillcraft-studio/rabota-s-skillcraft-studio) перед работой с SkillCraft Studio",
            ephemeral=True)

    @app_commands.command(name="new-prompt", description="Создает новый промпт")
    async def new_prompt(self, interaction: discord.Interaction, name: str, text: str = ""):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/new-prompt` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        user_folder_path = f"users/{user_id}"
        os.makedirs(user_folder_path, exist_ok=True)
        file_path = f"{user_folder_path}/{name}.txt"

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>")
            return

        if os.path.exists(file_path):
            await interaction.response.send_message("Ошибка: Промпт с таким именем уже существует.")
            return

        with open(file_path, 'w') as f:
            f.write(text)

        await interaction.response.send_message("Промпт успешно создан.")

    @app_commands.command(name="activate-key", description="Активировать OpenAI API ключ")
    @app_commands.choices(apply=[
        app_commands.Choice(name="Да", value="YES"),
        app_commands.Choice(name="Нет", value="NO"),
    ])
    async def activate_key(self, interaction: discord.Interaction, code: str, apply: str):
        await interaction.response.defer()
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/activate-key` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)

        with open("codes.txt", "r") as codes_file, open("keys.txt", "r") as keys_file:
            codes = codes_file.read().splitlines()
            keys = keys_file.read().splitlines()

        if code in codes:
            codes.remove(code)
            if apply == "YES" and keys:
                user_folder = f"users/{interaction.user.id}"
                if not os.path.exists(user_folder):
                    await interaction.followup.send("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>", ephemeral=True)
                    return
                selected_key = random.choice(keys)

                user_key_file_path = f"users/{interaction.user.id}/openai/key.txt"
                with open(user_key_file_path, "w") as user_key_file:
                    user_key_file.write(selected_key)

                embed = discord.Embed(title="Покупка OpenAI API ключа", color=discord.Color.green())
                embed.add_field(name="Покупка успешно завершена.", value=f"Ваш API ключ: **||{selected_key}||**\nAPI ключ был автоматически заменен")
                await interaction.followup.send(embed=embed, ephemeral=True)

                keys.remove(selected_key)

            elif apply == "NO" and keys:
                selected_key = random.choice(keys)

                embed = discord.Embed(title="Успешная покупка API ключа", color=discord.Color.green())
                embed.add_field(name="Покупка успешно завершена.", value=f"Ваш API ключ: **||{selected_key}||**")
                await interaction.followup.send(embed=embed, ephemeral=True)

                keys.remove(selected_key)

            else:
                await interaction.followup.send(
                    "Ошибка: API ключи закончились. Попробуйте повторить попытку чуть позже.", ephemeral=True)

            with open("codes.txt", "w") as codes_file, open("keys.txt", "w") as keys_file:
                codes_file.write("\n".join(codes))
                keys_file.write("\n".join(keys))
        else:
            await interaction.followup.send("Ошибка: Введенный код активации не существует.", ephemeral=True)

    @app_commands.command(name="edit-prompt", description="Редактирует промпт")
    async def edit_prompt(self, interaction: discord.Interaction, prompt_name: str, new_name: str = "", text: str = ""):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/edit-prompt` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        user_folder_path = f"users/{user_id}"
        file_path = f"{user_folder_path}/{prompt_name}.txt"
        await interaction.response.defer()

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>")
            return

        if not os.path.exists(file_path):
            await interaction.response.send_message("Ошибка: Промпт не найден.")
            return

        if not new_name and not text:
            await interaction.response.send_message("Ошибка: Для изменения промпта необходимо заполнить хотя бы одно редакционное поле.")
            return

        if new_name:
            new_file_path = f"{user_folder_path}/{new_name}.txt"

            if os.path.exists(new_file_path):
                await interaction.response.send_message("Ошибка: Файл с новым именем уже существует.")
                return

            os.rename(file_path, new_file_path)
            file_path = new_file_path

        if text:
            with open(file_path, 'w') as f:
                f.write(text)

        await interaction.response.send_message("Промпт успешно отредактирован.")

    @app_commands.command(name="prompts-list", description="Выводит список промптов")
    async def prompts_list(self, interaction: discord.Interaction):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/prompt-list` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)

        user_id = str(interaction.user.id)
        user_folder_path = f"users/{user_id}"

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:{command_id}>")
            return

        if not os.path.exists(user_folder_path):
            await interaction.response.send_message("Ошибка: Папка пользователя не найдена.")
            return

        files = [file for file in os.listdir(user_folder_path) if os.path.isfile(os.path.join(user_folder_path, file))]

        if not files:
            await interaction.response.send_message("Список промптов пуст.")
            return

        prompt_names = [os.path.splitext(file)[0] for file in files]

        embed = discord.Embed(title="Список промптов", description="\n".join(prompt_names), color=discord.Color.green())
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="delete-prompt", description="Удаляет промпт")
    async def delete_prompt(self, interaction: discord.Interaction, prompt_name: str):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/delete-prompt` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        user_folder_path = f"users/{user_id}"
        file_path = f"{user_folder_path}/{prompt_name}.txt"

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>")
            return

        if not os.path.exists(file_path):
            await interaction.response.send_message("Ошибка: Промпт не найден.")
            return

        os.remove(file_path)
        await interaction.response.send_message(f"Промпт `{prompt_name}` успешно удален.")

    @app_commands.command(name="change-key", description="Изменяет API-ключ")
    async def change_key(self, interaction: discord.Interaction, new_key: str):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/change-key` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        user_folder_path = f"users/{user_id}"
        openai_folder_path = f"{user_folder_path}/openai"
        key_file_path = f"{openai_folder_path}/key.txt"

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message(
                "Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>")
            return

        if not new_key.startswith("sk-") or len(new_key) > 75:
            await interaction.response.send_message("Ошибка: Некорректный новый API-ключ.")
            return

        if not os.path.exists(openai_folder_path):
            await interaction.response.send_message("Ошибка: Невозможно поменять api ключ.", ephemeral=True)
            return

        with open(key_file_path, 'w') as f:
            f.write(new_key)

        await interaction.response.send_message("API-ключ успешно изменен.", ephemeral=True)

    @app_commands.command(name="show-prompt", description="Показывает содержимое промпта")
    async def show_prompt(self, interaction: discord.Interaction, prompt_name: str):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/show-prompt` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        user_folder_path = f"users/{user_id}"
        file_path = f"{user_folder_path}/{prompt_name}.txt"

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>")
            return

        if not os.path.exists(file_path):
            await interaction.response.send_message("Ошибка: Промпт не найден или он пустой.")
            return

        with open(file_path, 'r') as f:
            prompt_content = f.read()

        max_chars_per_embed = 1024
        chunks = [prompt_content[i:i + max_chars_per_embed] for i in range(0, len(prompt_content), max_chars_per_embed)]

        for index, chunk in enumerate(chunks):
            embed = discord.Embed(title=f"Ваш промпт: {prompt_name} часть {index + 1})", color=discord.Color.blue())
            embed.add_field(name="Содержимое", value=chunk, inline=False)
            await interaction.response.send_message(embed=embed)

    @app_commands.command(name="test-prompt", description="Запускает исполнение промпта")
    async def test_prompt(self, interaction: discord.Interaction, prompt_name: str):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/test-prompt` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        user_folder_path = f"users/{user_id}"
        prompt_file_path = f"{user_folder_path}/{prompt_name}.txt"
        temporary_prompt_file_path = f"{user_folder_path}/openai/temporary_prompt.txt"

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>")
            return

        if not os.path.exists(prompt_file_path):
            await interaction.response.send_message("Ошибка: Промпт не найден.")
            return

        with open(prompt_file_path, 'r') as f:
            prompt_text = f.read()

        with open(temporary_prompt_file_path, 'w') as f:
            f.write(prompt_text)

        if not prompt_text.strip():
            await interaction.response.send_message("Ошибка: Промпт не содержит текста.")
            return

        await interaction.response.send_message(f"Промпт `{prompt_name}` был запущен.")

    @app_commands.command(name="test-chat", description="Чат с промптом")
    async def test_chat(self, interaction: discord.Interaction, message: str):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/test-chat` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        user_folder_path = f"users/{user_id}"
        openai_folder_path = f"{user_folder_path}/openai"
        key_file_path = f"{openai_folder_path}/key.txt"
        temporary_prompt_file_path = f"{openai_folder_path}/temporary_prompt.txt"
        await interaction.response.defer()

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>")
            return

        if not os.path.exists(openai_folder_path):
            await interaction.response.send_message("Ошибка: Невозможно вести диалог.")
            return

        with open(key_file_path, 'r') as f:
            api_key = f.read().strip()

        if not os.path.exists(temporary_prompt_file_path):
            await interaction.response.send_message("Ошибка: Невозможно вести диалог.")
            return

        with open(temporary_prompt_file_path, 'r') as f:
            temporary_prompt = f.read().strip()

        now = datetime.now()
        expiration_time = now + timedelta(minutes=120)
        if user_id not in context_store:
            context_store[user_id] = {
                "expiration_time": expiration_time,
                "context": []
            }
        else:
            if now > context_store[user_id]["expiration_time"]:
                context_store[user_id] = {
                    "expiration_time": expiration_time,
                    "context": []
                }

        context = context_store[user_id]["context"]
        context.append({"role": "user", "content": message})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": " "},
                          {"role": "user", "content": temporary_prompt}] + context,
                api_key=api_key
            )

            reply = response.choices[0].message.content.strip()
            respone = f"**Aika AI:** {reply}"
            context.append({"role": "assistant", "content": reply})

            await interaction.followup.send(respone)
        except Exception as e:
            await interaction.followup.send(f"При обработке запроса в чат произошла ошибка. Скорее всего из за отсутствия токенов. Купить новый API ключ помжно по команде `/buy-key`.")

    @app_commands.command(name="test-stop", description="Остановить тестовый чат")
    async def test_stop(self, interaction: discord.Interaction):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/test-stop` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        user_folder_path = f"users/{user_id}"
        openai_folder_path = f"{user_folder_path}/openai"
        temporary_prompt_file_path = f"{openai_folder_path}/temporary_prompt.txt"

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>")
            return

        if not os.path.exists(openai_folder_path):
            await interaction.response.send_message("Ошибка: Невозможно запустить промпт.")
            return

        if not os.path.exists(temporary_prompt_file_path):
            await interaction.response.send_message("Ошибка: Невозможно запустить промпт.")
            return

        with open(temporary_prompt_file_path, 'w') as f:
            f.write('')

        context_store.pop(user_id, None)

        await interaction.response.send_message("Выполнение промпта было остановлено.")

    @app_commands.command(name="profile", description="Показать профиль пользователя")
    async def show_profile(self, interaction: discord.Interaction):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/profile` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        users_file_path = 'users.txt'
        user_folder_path = f'users/{user_id}'
        openai_folder_path = f'{user_folder_path}/openai'
        key_file_path = f'{openai_folder_path}/key.txt'

        with open('users.txt', 'r') as f:
            register = [line.strip().split('#')[0] for line in f]
        if str(interaction.user.id) not in register:
            await interaction.response.send_message("Вы еще не зарегистрировались в SkillCraft Studio. Чтобы это сделать, воспользуйтесь командой </register:1131239719263547502>")
            return

        with open(users_file_path, 'r') as f:
            user_data = None
            for line in f:
                if line.startswith(f'{user_id}#'):
                    user_data = line.strip()
                    break

        if not user_data:
            await interaction.response.send_message("Ошибка: Вы не зарегистрированы.")
            return

        username = user_data.split("#")[1]

        api_key = open(key_file_path, 'r').read()

        if len(api_key) > 6:
            api_key = f"{api_key[:3]}{'*' * (len(api_key) - 6)}{api_key[-3:]}"

        prompt_count = len([name for name in os.listdir(user_folder_path) if name.endswith('.txt')])

        registration_date = user_data.split("#", 2)[-1].strip() if "#" in user_data else "Дата регистрации неизвестна"

        embed = discord.Embed(title=f"Профиль пользователя: {interaction.user.name}", color=discord.Color.blue())
        embed.set_thumbnail(url=interaction.user.avatar.url)
        embed.add_field(name="Никнейм", value=f"<@{user_id.split('#')[0]}>", inline=False)
        embed.add_field(name="API ключ", value=f"{api_key}\n> Купите API ключ всего за 20 рублей по команде `/buy-key`", inline=False)
        embed.add_field(name="ID пользователя", value=user_id, inline=False)
        embed.add_field(name="Кол-во промптов", value=prompt_count, inline=False)
        embed.add_field(name="Дата регистрации", value=registration_date, inline=False)

        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="buy-key", description="Купить API ключ OpenAI")
    async def buy_api_key(self, interaction: discord.Interaction):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/buy-key` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        embed = discord.Embed(title="Купить API ключ OpenAI", description="API ключ позволит вам начать использование SkillCraft Studio, а также даст возможность полноценного взаимодействия.", color=discord.Color.blue())
        embed.add_field(name="Купить API ключ", value="[Купить здесь](https://www.donationalerts.com/r/skillcraftstudio)", inline=False)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="info", description="Получить информацию о боте")
    async def show_info(self, interaction: discord.Interaction):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/info` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        version = "1.0.00 (release)"
        status = "🟢 - В полном порядке"
        ping = f"{round(self.bot.latency * 1000)}ms"
        users_file_path = 'users.txt'
        servers_count = len(self.bot.guilds)
        last_update_date = "<t:1691692620:D>, <t:1691692620:R>"

        with open(users_file_path, 'r') as f:
            users_count = len(f.readlines())

        embed = discord.Embed(title="Информация о боте", color=discord.Color.green())
        embed.add_field(name="Версия", value=version, inline=False)
        embed.add_field(name="Статус", value=status, inline=False)
        embed.add_field(name="Пинг", value=ping, inline=False)
        embed.add_field(name="Кол-во пользователей", value=str(users_count), inline=False)
        embed.add_field(name="Кол-во серверов", value=str(servers_count), inline=False)
        embed.add_field(name="Последнее обновление", value=last_update_date, inline=False)
        embed.add_field(name="Прочая информация", value="**[Политика Конфиденциальности](https://example.com/privacy) [Условия использования](https://example.com/terms)\n[Сервер поддержки](https://discord.gg/KKzBPg6jnu) [Документация](https://internet-2.gitbook.io/kaze-docs/skillcraft-studio/rabota-s-skillcraft-studio)**", inline=False)

        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="public-skill", description="Публикует навык")
    async def public_skill(self, interaction: discord.Interaction, name: str, logo: str, phrase_activate: str, short_describe: str, full_describe: str, tags: str):
        user = interaction.user.name
        message = (f'Пользователь {user} использовал команду `/public-skill` в канале `{interaction.channel.name if isinstance(interaction.channel, discord.TextChannel) else "Direct Message"}`')
        channel_id =
        channel = self.bot.get_channel(channel_id)
        await channel.send(message)
        user_id = str(interaction.user.id)
        user_folder_path = f'users/{user_id}'

        if not os.path.exists(user_folder_path):
            await interaction.response.send_message("Ошибка: Вы не зарегистрированы. Используйте команду /register.")
            return

        skill_file_path = f'{user_folder_path}/{name}.txt'
        if not os.path.exists(skill_file_path):
            await interaction.response.send_message("Ошибка: Навык с таким названием не найден.")
            return

        channel_id =
        channel = self.bot.get_channel(channel_id)

        user_embed = discord.Embed(title=f"Заявка на добавление навыка: {name}", color=discord.Color.blue())
        user_embed.add_field(name="ID Создателя", value=user_id, inline=False)
        user_embed.add_field(name="Название навыка", value=name, inline=False)
        user_embed.add_field(name="Лого навыка", value=logo, inline=False)
        user_embed.add_field(name="Фраза активатор навыка", value=phrase_activate, inline=False)
        user_embed.add_field(name="Краткое описание", value=short_describe, inline=False)
        user_embed.add_field(name="Полное описание", value=full_describe, inline=False)
        user_embed.add_field(name="Теги", value=tags, inline=False)

        await channel.send("Новый навык был отправлен на проверку @everyone", embed=user_embed)

        with open(skill_file_path, 'r') as f:
            skill_content = f.read()

        skill_embed = discord.Embed(title=f"Навык: {name}", description=skill_content, color=discord.Color.green())
        await channel.send(embed=skill_embed)

        await interaction.response.send_message(f"Навык `{name}` был отправлен на модерацию.")


async def setup(bot):
    await bot.add_cog(Slash(bot))