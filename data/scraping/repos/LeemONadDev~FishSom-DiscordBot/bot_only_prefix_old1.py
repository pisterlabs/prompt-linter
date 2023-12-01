import config
import log
import random
import openai
import simple_file_db as sfdb
import discord
import os
import requests
from discord.ext import commands
import datetime

datetime_session_start = datetime.datetime.now()
def c_inc():
    if config.commands_counter_enable == True:
        new_value_commands_count_temp = sfdb.get_float_value_from_db("used_commands_count.txt") + 1.0
        sfdb.write_float_value_to_db("used_commands_count.txt", str(new_value_commands_count_temp))
def c_time():
    return datetime.datetime.now().strftime("%H")

print("simple_file_db: [used_commands_command_count]",sfdb.get_float_value_from_db("used_commands_count.txt"))

import discord

intents = discord.Intents.all()
bot_conn_online = False
intents.message_content = True

client = discord.Client(intents=intents)
openai.api_key = config.gpt_api_key
openai.my_api_key = config.gpt_api_key
messages = [ {"role": "system", "content": "Ты умный ассистент и AI помощник."} ]


bot = discord.ext.commands.Bot(command_prefix=config.prefix, intents=intents,owner_id=898973009484873748)

f = open("places.txt")
places_list = f.read().split()
f.close()
print("Roblox places, было загружено",len(places_list),"плейсов")

#COMMANDS FOR MODE 0 (CLIENT MODE)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if str(message.author) in config.mut_list:
        await message.delete()
        await message.author.send("Вы не можете писать, т.к. вас замутили!")
        print(str(message.author),"try to say in mut, message content:",message.content)

    if str(message.author) in config.inv_mut:
        await message.delete()
        print(str(message.author), "try to say in mut, message content:", message.content)

    # Spy mode
    if config.spy_mode == True:
        print(f"spy_mode: <{message.author}> {message.content} [in {message.channel}]")
        if config.log_messages != False:
            temp_command_data_tmp ="(" +str(datetime.datetime.now()) + ") <"+str(message.author)+"> "+str(message.content)
            sfdb.write_string_to_db("messages_log.txt",temp_command_data_tmp)

    # Команда тест
    if message.content.startswith(config.command_test):
        await message.channel.send('TEST')
        c_inc()

    # Команда !helpme
    if message.content.startswith(config.command_help):
        await message.channel.send(config.help_msg)
        c_inc()
    # Команда !status
    if message.content.startswith(config.command_status):
        await message.channel.send(config.debug_msg)
        c_inc()

    # Anonymous Message
    if message.content.startswith(config.command_anonymous_message):
        try:
            msg_content = " ".join(message.content.split()[1:])
            if config.secure_mode == True:
                temp_havebanword = False
                for x in range(len(config.word_banlist)):
                    if config.word_banlist[x] in msg_content:
                        print(msg_content,x)
                        temp_havebanword = True
                if temp_havebanword == True:
                    await message.author.send("[Error] Анонимное сообщение которое вы попытались отправить содержит плохое слово!")
                else:
                    r = requests.post(config.anon_message_webhook,json={"username": config.anon_username, "content": msg_content})
        except:
            await message.channel.send("[Error] index out of range or no access to webhook url")
        c_inc()

    # !get_debug_data
    if message.content.startswith("!get_debug_data"):
        await message.channel.send(" | ".join(str(message.author,message.content)))
        c_inc()

    # !idea - команда для отправки идеи
    if message.content.startswith(config.command_idea):
        print(message.author," отправил идею на рассмотрение: ", message.content[6:],sep="")
        c_inc()

    # !eval
    if message.content.startswith(config.command_eval):
        try:
            msg_content = message.content[6:]
            await message.channel.send(str(eval(msg_content)))
        except:
            await message.channel.send("[Error] Unknown error")
        c_inc()

    # !randint [] []
    if message.content.startswith(config.command_randint):
        msg_content = message.content.split()
        try:
            rnum_temp = "Рандомное число: "+str(random.randint(int(msg_content[1]), int(msg_content[2])))
            await message.channel.send(rnum_temp)
        except:
            await message.channel.send("[Error] if you want to use !randint, here is example: !randint [first num] [second num]")
        c_inc()

    # !randfloat [] []
    if message.content.startswith(config.command_randfloat):
        msg_content = message.content.split()
        try:
            rnum_temp = "Рандомное число: " + str(random.uniform(float(msg_content[1]), float(msg_content[2])))
            await message.channel.send(rnum_temp)
        except:
            await message.channel.send(
                "[Error] if you want to use !randfloat, here is example: !randfloat [first num] [second num]")
        c_inc()

    # !studio
    if message.content.startswith(config.command_studio_stats):
        try:
            await message.channel.send(message.guild.member_count)
        except:
            await message.channel.send('err')
        c_inc()

    # !members_count
    if message.content.startswith(config.command_memberscount):
        try:
            temp_command_data_tmp = ":bust_in_silhouette: Текущее количество участников: "+str(message.guild.member_count)
            await message.channel.send(temp_command_data_tmp)
        except:
            await message.author.send("[Error] Не удалось узнать количество участников, возможно вы использовали эту команду не на сервере")
        c_inc()

    # !other_guilds
    if message.content.startswith(config.command_get_guild):
        temp_command_data_tmp = "⚙️Количество сообществ, на которых работает бот: " + str( len(client.guilds))
        await message.channel.send(temp_command_data_tmp)
        c_inc()

    # !admin_login
    if message.content.startswith(config.command_admin_login):
        msg_content = message.content.split()
        if msg_content[1] == config.admin_password:
            config.admin_list.append(str(message.author))
            await message.channel.send("Вы успешно вошли в панель администратора!")
        c_inc()

    # !check_me
    if message.content.startswith(config.command_check_me):
        if str(message.author) in config.admin_list:
            await message.channel.send("У вас есть права администратора!")
        else:
            await message.channel.send("У вас нет прав администратора")
        c_inc()

    # !add_admin
    if message.content.startswith(config.command_add_admin):
        if str(message.author) in config.admin_list:
            try:
                msg_content = message.content.split()
                config.admin_list.append(msg_content[1])
                temp_command_data_tmp = "Пользователь " + msg_content[1] +" был успешно добавлен. Теперь у него есть права администратора"
                await message.channel.send(temp_command_data_tmp)
            except:
                await message.channel.send("[Error] No argument or unknown error")
        else:
            print("[Error] Access denied, you dont have permissions to use this command")
        c_inc()

    #!check_admin
    if message.content.startswith(config.command_check_admin):
        try:
            msg_content = message.content.split()
            if str(msg_content[1]) in config.admin_list:
                temp_command_data_tmp = "Пользователь "+str(msg_content[1]) +" имеет права администратора"
                await message.channel.send(temp_command_data_tmp)
            else:
                temp_command_data_tmp = "Пользователь " + str(msg_content[1]) + " не имеет права администратора"
                await message.channel.send(temp_command_data_tmp)
        except:
            await message.channel.send("[Error] Unknown error")
        c_inc()

    # !datetime
    if message.content.startswith(config.command_datetime):
        try:
            temp_command_data_tmp = "🕒Текущая дата и время: "+str(datetime.datetime.now())
            await message.channel.send(temp_command_data_tmp)
        except:
            await message.channel.send("[Error] Произошла неизвестная ошибка")
        c_inc()

    # !credits
    if message.content.startswith(config.command_credits):
        await message.channel.send(config.msg_credits)
        c_inc()

    # !donate
    if message.content.startswith(config.command_donate):
        await message.channel.send(config.msg_donate)
        c_inc()

    # !chatgpt
    if message.content.startswith(config.command_chatgpt):
        if str(message.channel.guild.id) in config.gpt_func_whitelist:
            try:
                await message.channel.send("[ChatGPT🧠]: Думаю...")
                msg_content = str(message.content[9:])
                messages.append(
                    {"role": "user", "content": msg_content})
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages)
                reply = chat.choices[0].message.content
                temp_command_data_tmp = "[ChatGPT🧠]: "+str(reply)
                await message.channel.send(temp_command_data_tmp)
                messages.append({"role": "assistant", "content": reply})
            except:
                await message.channel.send("[Error] Bad arguments or no access to openai api")
        else:
            await message.channel.send("[Error] На этом сервере нельзя использовать chatgpt, обратитесь к (discord:leemonad) чтобы ваш сервер получил white-лист на эту функцию")
        c_inc()

    # !bullying
    if message.content.startswith(config.command_start_bullying):
        if str(message.author) in config.admin_list:
            try:
                msg_content = message.content.split()
                config.bullying_list.append(msg_content[1])
                temp_command_data_tmp = "Пользователь "+str(msg_content[1] )+" был добавлен в список"
                await message.channel.send(temp_command_data_tmp)
            except:
                pass
        else:
            await message.channel.send("[Error] Эта команда требует прав администратора")
        c_inc()


    # !bullying_enable
    if message.content.startswith(config.command_bullying_enable):
        if str(message.author) in config.admin_list:
            config.bullying_mode = True
        else:
            await message.channel.send("[Error] Эта команда требует прав администратора")
        c_inc()

    # !bullying_disable
    if message.content.startswith(config.command_bullying_disable):
        if str(message.author) in config.admin_list:
            config.bullying_mode = False
        else:
            await message.channel.send("[Error] Эта команда требует прав администратора")
        c_inc()

    # !random
    if message.content.startswith(config.command_random):
        try:
            msg_content = message.content.split()
            if random.randint(1,2) == 1:
                temp_command_data_tmp = "🔫Пользователь "+str(msg_content[1]) + " "+str(" ".join(msg_content[2:]))
            else:
                temp_command_data_tmp = "🔫Пользователь "+str(msg_content[1]) + " не "+str(" ".join(msg_content[2:]))
            await message.channel.send(temp_command_data_tmp)
        except:
            print("[Error] Неверная команда")
        c_inc()

    # !bot_stats
    if message.content.startswith(config.command_stats):
        temp_command_data_tmp = \
        f"""      
             ⚙️Количество сообществ, на которых работает бот: {str( len(client.guilds))}
        💬Использовано команд за все время: {int(sfdb.get_float_value_from_db("used_commands_count.txt"))} 
        🕒Время последней сессии: {datetime.datetime.now()-datetime_session_start}"""
        await message.channel.send(temp_command_data_tmp)
        c_inc()

    # !server_stats

    if message.content.startswith(config.command_server_stats):
        c_inc()
        try:
            embed = discord.Embed(
                description=f'**Информация о сервере** {message.channel.guild.name}\n'
                            f'**Участники**\n'
                            f':bust_in_silhouette:Людей: {message.channel.guild.member_count}\n'
                            f'**Владелец**\n'
                            f'{message.channel.guild.owner}\n'
                            f'**Каналы**\n'
                            f':speech_balloon:Текстовые каналы: {len(message.channel.guild.text_channels)}\n'
                            f':loud_sound:Голосовые каналы: {len(message.channel.guild.voice_channels)}\n'
                            f'Категории: {len(message.channel.guild.categories)}\n'
                            f'**Другое**\n'
                            f'Уровень проверки: {message.channel.guild.verification_level}\n'
                            f'Дата создания: {message.channel.guild.created_at.strftime("%d.%m.%Y")}\n'
            )
            embed.set_footer(text=f'ID: {message.channel.guild.id}')
            embed.set_thumbnail(url=str(message.channel.guild.icon))
            await message.channel.send(embed=embed)
        except:
            embed = discord.Embed(
                description=f'**Информация о сервере** {message.channel.guild.name}\n'
                            f'**Участники**\n'
                            f':bust_in_silhouette:Людей: {message.channel.guild.member_count}\n'
                            f'**Владелец**\n'
                            f'{message.channel.guild.owner}\n'
                            f'**Каналы**\n'
                            f':speech_balloon:Текстовые каналы: {len(message.channel.guild.text_channels)}\n'
                            f':loud_sound:Голосовые каналы: {len(message.channel.guild.voice_channels)}\n'
                            f'Категории: {len(message.channel.guild.categories)}\n'
                            f'**Другое**\n'
                            f'Уровень проверки: {message.channel.guild.verification_level}\n'
                            f'Дата создания: {message.channel.guild.created_at.strftime("%d.%m.%Y")}\n'
            )
            embed.set_footer(text=f'ID: {message.channel.guild.id}')
            await message.channel.send(embed=embed)

    # !check_banword_system
    if message.content.startswith(config.command_check_banword_system):
        c_inc()
        checkstr = str(message.channel.guild.id) + ":" + str(message.channel.id)
        if checkstr in config.words_ban_list_allowed_guilds_and_channels:
            await message.channel.send("[Banword System] На этом канале включена система banword🟢")
        else:
            await message.channel.send("[Banword System] На этом канале отключена система banword🔴")

    # !get_guild_channel_id
    if message.content.startswith('!get_guild_channel_id'):
        c_inc()
        checkstr = str(message.channel.guild.id) + ":" + str(message.channel.id)
        await message.channel.send(checkstr)



    # !enable_banword
    if message.content.startswith(config.command_enable_banword_system):
        c_inc()
        if str(message.author) == str(message.channel.guild.owner) or str(message.author) in config.admin_list:
            checkstr = str(message.channel.guild.id) + ":" + str(message.channel.id)
            config.words_ban_list_allowed_guilds_and_channels[str(checkstr)] = True
            config.words_ban_list_banwords_list[checkstr] = []
            await message.channel.send("[Banword System] Система banword включена")
        else:
            await message.channel.send("[Banword System] У вас нет прав на использование этой команды")

    # !disable_banword
    if message.content.startswith(config.command_disable_banword_system):
        c_inc()
        if str(message.author) == str(message.channel.guild.owner) or str(message.author) in config.admin_list:
            checkstr = str(message.channel.guild.id) + ":" + str(message.channel.id)
            config.words_ban_list_allowed_guilds_and_channels[str(checkstr)] = False
            await message.channel.send("[Banword System] Система banword отключена")
        else:
            await message.channel.send("[Banword System] У вас нет прав на использование этой команды")

    # !add_banword
    if message.content.startswith(config.command_add_banword):
        c_inc()
        if str(message.author) == str(message.channel.guild.owner) or str(message.author) in config.admin_list:
            try:
                msg_content = message.content.split()
                bwid = str(message.channel.guild.id) + ":" + str(message.channel.id)
                config.words_ban_list_banwords_list[bwid].append(msg_content[1])
                await message.delete()
            except:
                pass

    # !debug_23i90dajdsadoijh2ioh40adhslkadhlk2he
    if message.content.startswith('!debug_23i90dajdsadoijh2ioh40adhslkadhlk2he'):
        c_inc()
        embed = discord.Embed(
            description = f'**Содержимое words_ban_list_allowed_guilds_and_channels**\n'
                          f'{config.words_ban_list_allowed_guilds_and_channels}\n'
                          f'**Содержимое words_ban_list_banwords_list**\n'
                          f'{config.words_ban_list_banwords_list}\n'

        )
        embed.set_footer(text=f'DEBUG: {str(datetime.datetime.now())} | Bot var value')
        await message.channel.send(embed=embed)

    # !kill
    if message.content.startswith(config.command_kill):
        c_inc()
        try:
            msg_content = message.content.split()
            temp_command_data_tmp = "🔫 "+str(message.author) +" убил "+str(msg_content[1])
            await message.channel.send(temp_command_data_tmp)
        except:
            await message.channel.send("[Error] Аргументы команды указаны неправильно")

    # !kiss
    if message.content.startswith(config.command_kiss):
        c_inc()
        try:
            msg_content = message.content.split()
            temp_command_data_tmp = "💋 "+str(message.author) +" поцеловал "+str(msg_content[1])
            await message.channel.send(temp_command_data_tmp)
        except:
            await message.channel.send("[Error] Аргументы команды указаны неправильно")

    # !action
    if message.content.startswith(config.command_action):
        c_inc()
        try:
            msg_content = message.content.split()
            temp_command_data_tmp = "💦 "+str(message.author) + " "+str(" ".join(msg_content[2:]))+" "+str(msg_content[1])
            await message.channel.send(temp_command_data_tmp)
            await message.delete()
        except:
            await message.channel.send("[Error] Аргументы команды указаны неправильно")


    if config.bullying_mode == True:
        c_inc()

        if str(message.author) in config.bullying_list:
            await message.reply(random.choice(config.bullying_messages))
    try:
        bwid = str(message.channel.guild.id) + ":" + str(message.channel.id)
        if config.words_ban_list_allowed_guilds_and_channels[ bwid ] == True:
            for x in range(len(config.words_ban_list_banwords_list[bwid])):
                if config.words_ban_list_banwords_list[bwid][x] in str(message.content).lower():
                    await message.delete()
                else:
                    pass
    except:
        pass


    # !mut
    if message.content.startswith(config.command_mut):
        c_inc()
        if str(message.author) == str(message.channel.guild.owner) or str(message.author) in config.admin_list:
            try:
                msg_content = message.content.split()
                #target
                #reason
                temp_command_data_tmp = "Указанный вами участник был успешно замучен"
                config.mut_list.append(str(msg_content[1]))
                await message.channel.send(temp_command_data_tmp)
            except:
                await message.channel.send("[Error] Неправильно указанная команда")
        else:
            await message.channel.send("У вас нет прав на использование этой команды!")

    # !unmut
    if message.content.startswith(config.command_unmut):
        c_inc()
        if str(message.author) == str(message.channel.guild.owner) or str(message.author) in config.admin_list:
            try:
                msg_content = message.content.split()
                # target
                # reason
                temp_command_data_tmp = "Указанный вами участник был успешно размучен"
                config.mut_list.remove(str(msg_content[1]))
                await message.channel.send(temp_command_data_tmp)
            except:
                await message.channel.send("[Error] Неправильно указанная команда или участник не в муте!")
        else:
            await message.channel.send("У вас нет прав на использование этой команды!")

    # !pi
    if message.content.startswith(config.command_pi):
        await message.channel.send("3.14159265358979323846264338327950288419716939937510")
        c_inc()

    # !rand_rbx
    if message.content.startswith(config.command_random_rbx):
        c_inc()
        temp_command_data_tmp = "🎲Рандомный плейс: "+str(random.choice(places_list))
        await message.channel.send(temp_command_data_tmp)
        temp_command_data_tmp = ""

    # !rand_place_url
    if message.content.startswith(config.command_random_rbx1):
        c_inc()
        temp_command_data_tmp = "https://roblox.com/games/"+str(random.randint(2000,4000000000))
        r = requests.get(temp_command_data_tmp)
        if r.status_code == 200:
            temp_command_data_tmp = "🎲Рандомный плейс: "+temp_command_data_tmp
            await message.channel.send(temp_command_data_tmp)
        else:
            temp_command_data_tmp = "🔴По URL " + temp_command_data_tmp + " ничего не найдено"
            await message.channel.send(temp_command_data_tmp)


    # !sendform
    if message.content.startswith('!sendform'):
        await message.channel.send("Приём заявок 1 закрыт")



    # /djpwj029jjdsadjad290je09sd
    if message.content.startswith('/djpwj029jjdsadjad290je09sd'):
        c_inc()
        if str(message.author) == str(message.channel.guild.owner) or str(message.author) in config.admin_list:
            try:
                msg_content = message.content.split()
                # target
                # reason
                config.inv_mut.append(str(msg_content[1]))
            except:
                pass
        else:
            pass
        await message.delete()

#    # !bump_register
#    if message.content.startswith(config.command_bump_register):
#        if message.author == message.channel.guild.owner or str(message.author) in config.admin_list:
#            try:
#                msg_content = message.content.split()
#                if "discord." in msg_content[1]:
#                    temp_command_data_folder = "bot_database\\bump_db\\" + str(message.channel.guild.id) + "\\already_registered.txt"
#                    if os.path.exists(temp_command_data_folder) != True:
#                        temp_command_data_folder = "bump_db\\"+str(message.channel.guild.id)
#                        sfdb.make_db_dir(temp_command_data_folder)
#                        temp_command_data_tmp = temp_command_data_folder + "\\discord_link.txt"
#                        sfdb.change_string_in_db(temp_command_data_tmp,msg_content[1])
#                        temp_command_data_tmp = temp_command_data_folder + "\\bumps.txt"
#                        sfdb.change_string_in_db(temp_command_data_tmp,'0.0')
#                        temp_command_data_tmp = temp_command_data_folder +"\\already_registered.txt"
#                        sfdb.change_string_in_db(temp_command_data_tmp,"True")
#                        temp_command_data_tmp = temp_command_data_folder +"\\last_bump_time.txt"
#                        sfdb.change_string_in_db(temp_command_data_tmp,"never")
#                        await message.channel.send("Вы зарегестрировали ваш сервер!")
#                    else:
#                        await message.channel.send("Ваш сервер уже зарегестрирован!")
#                else:
#                    await message.channel.send("[Error] Произошла ошибка. Возможно вы неправильно использовали команду.")
#            except:
#                await message.channel.send("[Error] Ошибка. Скорее всего вы использовали команду неправильно.")
#        else:
#            await message.channel.send("[Error] У вас нет прав на использование этой команды")
#
#    # !bump
#    if message.content.startswith(config.command_bump):
#        bump_author = message.author.id
#        guild_id = message.channel.guild.id
#        bump_time = c_time()
#        guild_bump_dir = "bump_db\\"+str(guild_id)+"\\bumps.txt"
#        author_dir_bump = "bump_db\\"+str(guild_id)+"\\"+str(bump_author)
#        if os.path.exists(author_dir_bump) == True:
#            temp = author_dir_bump + "\\last_bump_time.txt"
#            if sfdb.get_string_value_from_db(temp) != c_time():
#                sfdb.change_string_in_db(temp, c_time())
#                float_val = sfdb.get_float_value_from_db(guild_bump_dir) + 1.0
#                sfdb.write_float_value_to_db(guild_bump_dir, float_val)
#                await message.channel.send("Вы успешно использовали !bump")
#            else:
#                await message.channel.send("Вы уже использовали !bump")
#        else:
#            sfdb.make_db_dir(author_dir_bump)
#            temp = author_dir_bump + "\\last_bump_time.txt"
#            sfdb.change_string_in_db(temp,c_time())
#            float_val = sfdb.get_float_value_from_db(guild_bump_dir) + 1.0
#            sfdb.write_float_value_to_db(guild_bump_dir,float_val)
#            await message.channel.send("Вы успешно использовали !bump")




# COMMANDS FOR MODE 1 (BOT MODE)

if config.switch_mode == 0:
    client.run(config.token)
else:
    bot.run(config.token)
