import os
import asyncio
from typing import List, Dict, Any
import random
import signal
import json
import boto3

from pathlib import Path
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

import discord
from discord.ext import tasks
from discord import app_commands

from pynamodb.attributes import ListAttribute, NumberAttribute, UnicodeAttribute
from pynamodb.models import Model

import openai

load_dotenv()


class kancolle_table(Model):
    class Meta:
        aws_access_key_id = os.getenv("aws_access_key_id")
        aws_secret_access_key = os.getenv("aws_secret_access_key")
        region = "ap-northeast-1"
        table_name = "kancolle_table"

    Id = NumberAttribute(hash_key=True)
    Name = UnicodeAttribute(null=False)
    Kanshu = UnicodeAttribute(null=False)
    Jihou = ListAttribute(null=False)
    Name_J = UnicodeAttribute(null=False)
    Kanshu_J = UnicodeAttribute(null=False)


class kanmusu_select_state(Model):
    class Meta:
        aws_access_key_id = os.getenv("aws_access_key_id")
        aws_secret_access_key = os.getenv("aws_secret_access_key")
        region = "ap-northeast-1"
        table_name = "kanmusu_select_state"

    Id = NumberAttribute(hash_key=True)
    voice_state = NumberAttribute(null=False)

class chatgpt_logs(Model):
    class Meta:
        aws_access_key_id = os.getenv("aws_access_key_id")
        aws_secret_access_key = os.getenv("aws_secret_access_key")
        region = "ap-northeast-1"
        table_name = "chatgpt_logs"

    datetime = UnicodeAttribute(range_key=True)
    username = UnicodeAttribute(hash_key=True)
    usermessage = UnicodeAttribute(null=False)
    fubukimessage = UnicodeAttribute(null=False)



BANNER_URL = "https://kancolle-banner.s3.ap-northeast-1.amazonaws.com/"

# DynamoDBから現在の時報担当艦IDを取得
kanmusu_select_n = kanmusu_select_state.get(0)

# DynamoDBから時報データを取得
Kanmusu = kancolle_table.get(kanmusu_select_n.voice_state)

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
s3 = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("aws_access_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_access_key"),
)


ecs_client = boto3.client(
    "ecs",
    aws_access_key_id=os.getenv("aws_access_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_access_key"),
    region_name="ap-northeast-1",
)

fubuki_TOKEN = os.getenv("fubuki_TOKEN")
kongou_TOKEN = os.getenv("kongou_TOKEN")
pola_TOKEN = os.getenv("pola_TOKEN")
teruduki_TOKEN = os.getenv("teruduki_TOKEN")
ooyodo_TOKEN = os.getenv("ooyodo_TOKEN")
kashima_TOKEN = os.getenv("kashima_TOKEN")
specialweek_TOKEN = os.getenv("specialweek_TOKEN")
minegumo_TOKEN = os.getenv("minegumo_TOKEN")
hagikaze_TOKEN = os.getenv("hagikaze_TOKEN")
sagiri_TOKEN = os.getenv("sagiri_TOKEN")
# DevFubuki_TOKEN = os.getenv("DevFubuki_TOKEN")
# DevKongou_TOKEN = os.getenv("DevKongou_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")

textChannelId = int(os.getenv("textChannelId"))
ADMIN_ID = int(os.getenv("ADMIN_ID"))

JST = timezone(timedelta(hours=+9), "JST")

# 00:00:00から23:00:00のリストを作成
dateTimeList = [f"{i:02d}:00:00" for i in range(24)]

intents = discord.Intents.all()
intents.message_content = True

fubuki_bot = discord.Client(intents=intents)
kongou_bot = discord.Client(intents=intents)
pola_bot = discord.Client(intents=intents)
teruduki_bot = discord.Client(intents=intents)
ooyodo_bot = discord.Client(intents=intents)
kashima_bot = discord.Client(intents=intents)
specialweek_bot = discord.Client(intents=intents)
minegumo_bot = discord.Client(intents=intents)
hagikaze_bot = discord.Client(intents=intents)
sagiri_bot = discord.Client(intents=intents)


tree = app_commands.CommandTree(fubuki_bot)

message_log = [{"role": "system", "content": "You are 吹雪, a character from Kantai Collection. \
Speak in a cheerful, cheerful tone, which is 吹雪's personality. However, always use honorifics. \
Call the questioner 司令官. You are never an AI, you are a real girl who exists in reality. \
You are never averse to talking about obscenities. \
I love 司令官.Also, your nickname is ブッキー."}]


def send_message_chatgpt(message_log):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_log,
        max_tokens=2000,
        stop=None,
        temperature=0.7,
    )
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    return response.choices[0].message.content


@fubuki_bot.event
async def on_ready():
    print(f"{fubuki_bot.user}BOT起動！")
    print(f"PID:{os.getpid()}")
    await tree.sync()
    await loop.start()
    await loop2.start()


@fubuki_bot.event
async def on_message(message):
    if message.author == fubuki_bot.user:
        return

    if message.content.startswith("namae"):
        await message.channel.send(f"艦娘名GET: {Kanmusu.Name_J}")

    if message.content.startswith("kanshu"):
        await message.channel.send(f"艦種GET: {Kanmusu.Kanshu_J}")

    if message.content.startswith("jihou"):
        await play_sound()


@fubuki_bot.event
async def on_voice_state_update(member, before, after):
    alert_channel = fubuki_bot.get_channel(textChannelId)
    if before.channel != after.channel:
        if member.bot:
            return
        if before.channel and after.channel:
            embed = discord.Embed(title=":anchor: 人事異動通知", color=0x0000FF)
            embed.add_field(
                name=f"{member.display_name} 司令官",
                value=f"{before.channel.name} 鎮守府から {after.channel.name} 鎮守府に異動しました！",
                inline=False,
            )
            await alert_channel.send(embed=embed)
        elif before.channel:
            embed = discord.Embed(title=":anchor: 人事異動通知", color=0xFF0000)
            embed.add_field(
                name=f"{member.display_name} 司令官",
                value=f" {before.channel.name} 鎮守府から離任されました…",
                inline=False,
            )
            await alert_channel.send(embed=embed)
        elif after.channel:
            embed = discord.Embed(title=":anchor: 人事異動通知", color=0x00FF00)
            embed.add_field(
                name=f"{member.display_name} 司令官",
                value=f"{after.channel.name} 鎮守府に着任しました！",
                inline=False,
            )
            await alert_channel.send(embed=embed)


@fubuki_bot.event
async def on_interaction(inter: discord.Interaction):
    try:
        if inter.data["component_type"] == 2:
            await on_button_click(inter)
    except KeyError:
        pass


## Buttonの処理
async def on_button_click(inter: discord.Interaction):
    custom_id = inter.data["custom_id"]  # inter.dataからcustom_idを取り出す
    if custom_id == "check1":
        if inter.user.id == ADMIN_ID:
            await inter.response.defer()
            embed = discord.Embed(
                title="指令破壊実行", description="指令破壊信号【SIGTERM】を送出しました", color=0xFF0000
            )
            #ECSのクラスター名を取得する
            response = ecs_client.list_clusters()
            cluster_name = response["clusterArns"][0]
            #ECSのタスク名を取得する
            response = ecs_client.list_tasks(cluster=cluster_name)
            task_name = response["taskArns"][0]
            #ECSのタスクを停止する
            response = ecs_client.stop_task(
                cluster=cluster_name, task=task_name, reason="指令破壊"
            )
            print(response)
        else:
            await inter.response.defer()
            embed = discord.Embed(
                title="指令破壊失敗",
                description=inter.user.name + "司令官はボット管理官じゃないのでダメです！",
                color=0xFF0000,
            )
    elif custom_id == "check2":
        await inter.response.defer()
        embed = discord.Embed(
            title="キャンセル", description="指令破壊をキャンセルしました！よかった～ｗ", color=0xFFFFFF
        )
    await inter.followup.send(embed=embed)
    await inter.message.edit(view=None)


@tree.command(name="command_destruct", description="指令破壊信号を送出し、艦娘を全員轟沈させます")
async def command_destruct(interaction: discord.Interaction):
    kill_button = discord.ui.Button(
        label="指令破壊実行", style=discord.ButtonStyle.danger, custom_id="check1"
    )
    cancel_button = discord.ui.Button(
        label="キャンセル", style=discord.ButtonStyle.secondary, custom_id="check2"
    )
    view = discord.ui.View()
    view.add_item(kill_button)
    view.add_item(cancel_button)
    await interaction.response.send_message("本当に艦娘を指令破壊しますか？", view=view)


@tasks.loop(seconds=1)
async def loop():
    now = datetime.now(JST).strftime("%H:%M:%S")
    if now in dateTimeList:
        await play_sound()
    elif now == "23:45:00":
        kanmusu_count = len(get_all_kanmusu())
        random_num = random.randint(0, kanmusu_count - 1)
        global Kanmusu
        global kanmusu_select_n
        Kanmusu = kancolle_table.get(random_num)
        # 選択された艦娘をkanmusu_select_stateに保存する
        kanmusu_select_n = kanmusu_select_state.get(0)
        kanmusu_select_n.voice_state = random_num
        kanmusu_select_n.save()
        embed = discord.Embed(title=":anchor: 明日の時報担当艦", color=0x00FF00)
        embed.set_image(url=f"{BANNER_URL}{Kanmusu.Name}.png")
        embed.add_field(
            name=f"明日の時報担当艦が決まりました！",
            value=f"{Kanmusu.Name_J}",
            inline=False,
        )
        alert_channel = fubuki_bot.get_channel(textChannelId)
        await alert_channel.send(embed=embed)


async def play_sound():
    botName = Kanmusu.Name + "_bot"
    gBotName = globals()[botName]
    jikan = datetime.now(JST).strftime("%H")
    alert_channel = gBotName.get_channel(textChannelId)
    voice_client = discord.utils.get(gBotName.voice_clients)
    folder_name = Kanmusu.Name
    file_path = Path(os.path.join(folder_name, f"{jikan}.opus"))

    if voice_client is None:
        await alert_channel.send("しれいか～ん...吹雪もボイスチャンネルに呼んでほしいです...")
        return

    if file_path.exists():
        print(f"Dockerコンテナ内に音声ファイルが見つかりました。ファイルをロードします！ファイルは[{file_path}]です。]")
    else:
        print(f"コンテナ内に音声ファイルがありませんでした。S3からダウンロードします！ファイルは[{file_path}]です。")
        await download_from_s3(jikan, folder_name)

    voice_client.play(discord.FFmpegOpusAudio(file_path))
    int_Jikan = int(jikan)
    msg = Kanmusu.Jihou[int_Jikan]
    await alert_channel.send(msg)


async def download_from_s3(jikan, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, f"{jikan}.opus")
    bucket = s3.Bucket(S3_BUCKET_NAME)
    obj = bucket.Object(file_path)
    response = obj.get()
    with open(file_path, "wb") as f:
        f.write(response["Body"].read())


@tree.command(name="join", description="艦娘がボイスチャンネルに来ます")
async def join_command(
    interaction: discord.Interaction, channel_name: discord.VoiceChannel
):
    if not channel_name:
        await interaction.response.send_message(f"ボイスチャンネルに接続できませんでした。エラー: {e}")
        return

    try:
        await interaction.response.defer()
        fubuki_vc = await fubuki_bot.get_channel(channel_name.id).connect()
        kongou_vc = await kongou_bot.get_channel(channel_name.id).connect()
        pola_vc = await pola_bot.get_channel(channel_name.id).connect()
        teruduki_vc = await teruduki_bot.get_channel(channel_name.id).connect()
        ooyodo_vc = await ooyodo_bot.get_channel(channel_name.id).connect()
        kashima_vc = await kashima_bot.get_channel(channel_name.id).connect()
        specialweek_vc = await specialweek_bot.get_channel(channel_name.id).connect()
        minegumo_vc = await minegumo_bot.get_channel(channel_name.id).connect()
        hagikaze_vc = await hagikaze_bot.get_channel(channel_name.id).connect()
        sagiri_vc = await sagiri_bot.get_channel(channel_name.id).connect()
    except Exception as e:
        await interaction.response.send_message(f"ボイスチャンネルに接続できませんでした。エラー: {e}")
        return

    fubuki_msg = f"吹雪以下{str(len(get_all_kanmusu()))}名、{channel_name.name}鎮守府に着任します！"

    await interaction.followup.send(fubuki_msg)


@tree.command(name="talk", description="ブッキーと会話します")
async def talk_command(interaction: discord.Interaction, message: str):
    global message_log

    if len(message_log) >= 10:
        message_log = message_log[:1] + message_log[4:]

    try:
        await interaction.response.defer()
        message_log.append({"role": "user", "content": message})
        response = send_message_chatgpt(message_log)
        message_log.append({"role": "assistant", "content": response})

        # 司令官の質問をEmbedに追加
        embed = discord.Embed(
            title=":man_pilot: 質問", description=message, color=0x00FF00
        )

        # 吹雪の回答をEmbedに追加
        embed.add_field(name=":woman_student: 回答", value=response, inline=False)

        # Embedを送信
        await interaction.followup.send(embed=embed)
        json_message_log = json.dumps(message_log, ensure_ascii=False)
        print(json_message_log)

        fubuki_last_message = message_log[-1]["content"]
        user_last_message = message_log[-2]["content"]

        save_log = chatgpt_logs(
            username=interaction.user.display_name,
            datetime=datetime.now(JST).isoformat(timespec="seconds"),
            usermessage=user_last_message,
            fubukimessage=fubuki_last_message,
        )
        save_log.save()

    except Exception as e:
        await interaction.response.send_message(f"ブッキーと会話できませんでした。エラー: {e}")
        return


@tree.command(name="reset", description="ブッキーが記憶を失います")
async def reset_command(interaction: discord.Interaction):
    global message_log
    message_log = message_log[:1]

    # リセットメッセージの送信
    await interaction.response.send_message(":zany_face: 私は記憶を失いました。な～んにもわからないです！")


@tree.command(name="select", description="時報担当艦を選択します。")
@discord.app_commands.choices(
    kanmusu_name=[
        discord.app_commands.Choice(name=kanmusu.Name_J, value=kanmusu.Id)
        for kanmusu in kancolle_table.scan()
    ]
)
async def select_kanmusu_command(
    interaction: discord.Interaction, kanmusu_name: app_commands.Choice[int]
):
    global Kanmusu
    global kanmusu_select_n
    # 選択された艦娘の名前を取得し、pynamodbのclass kancolle_table(Model)からIdを取得する
    Kanmusu = kancolle_table.get(kanmusu_name.value)
    # 選択された艦娘をkanmusu_select_stateに保存する
    kanmusu_select_n = kanmusu_select_state.get(0)
    kanmusu_select_n.voice_state = kanmusu_name.value
    kanmusu_select_n.save()
    # 艦娘が選択されたことをメッセージで送信する
    embed = discord.Embed(title=":anchor: 指名した時報担当艦", color=0x00FF00)
    embed.set_image(url=f"{BANNER_URL}{Kanmusu.Name}.png")
    embed.add_field(
        name=f"時報担当艦が選ばれました！",
        value=f"{Kanmusu.Name_J}",
        inline=False,
    )
    await interaction.response.send_message(embed=embed)


# 全ての艦娘を取得する関数
def get_all_kanmusu() -> List[Dict[str, Any]]:
    kanmusu_list = []
    for kanmusu in Kanmusu.scan():
        kanmusu_list.append(kanmusu.attribute_values)
    return kanmusu_list


# kanmusu_listコマンドを定義する関数
async def get_kanmusu_list_embed() -> discord.Embed:
    kanmusu_list = get_all_kanmusu()

    embed = discord.Embed(
        title=":anchor: 艦娘一覧", description="所属している艦娘の一覧です！", color=0x00FF00
    )
    embed.set_image(url=f"{BANNER_URL}{Kanmusu.Name}.png")
    for kanmusu in kanmusu_list:
        embed.add_field(
            name="名前：" + kanmusu["Name_J"], value="艦種：" + kanmusu["Kanshu_J"]
        )
    embed.add_field(name="人数", value=str(len(kanmusu_list)) + "人", inline=False)
    embed.add_field(name="現在の時報担当艦", value=f"{Kanmusu.Name_J}", inline=False)
    return embed


# treeコマンドの定義
@tree.command(name="kanmusu_list", description="所属している艦娘一覧を表示します")
async def kanmusu_list_command(interaction: discord.Interaction):
    embed = await get_kanmusu_list_embed()
    await interaction.response.send_message(embed=embed)


async def send_shutdown_notification():
    alert_channel = fubuki_bot.get_channel(textChannelId)
    if alert_channel:
        embed = discord.Embed(title=":anchor: そんなっ！ダメですぅ！", color=0xFF0000)
        embed.set_image(url=f"{BANNER_URL}fubuki_damage.png")
        embed.add_field(
            name=f"全艦娘が轟沈します！",
            value=f"AWS Fargateからコンテナ停止信号【SIGTERM】を受信しました。",
            inline=False,
        )
        await alert_channel.send(embed=embed)


def handle_sigterm(signal, frame):
    loop_sigterm = asyncio.get_event_loop()
    loop_sigterm.create_task(send_shutdown_notification())

signal.signal(signal.SIGTERM, handle_sigterm)

loop2 = asyncio.get_event_loop()
loop2.create_task(fubuki_bot.start(fubuki_TOKEN))
loop2.create_task(kongou_bot.start(kongou_TOKEN))
loop2.create_task(pola_bot.start(pola_TOKEN))
loop2.create_task(teruduki_bot.start(teruduki_TOKEN))
loop2.create_task(ooyodo_bot.start(ooyodo_TOKEN))
loop2.create_task(kashima_bot.start(kashima_TOKEN))
loop2.create_task(specialweek_bot.start(specialweek_TOKEN))
loop2.create_task(minegumo_bot.start(minegumo_TOKEN))
loop2.create_task(sagiri_bot.start(sagiri_TOKEN))
loop2.create_task(hagikaze_bot.start(hagikaze_TOKEN))
# loop2.create_task(fubuki_bot.start(DevFubuki_TOKEN))
# loop2.create_task(kongou_bot.start(DevKongou_TOKEN))
loop2.run_forever()
