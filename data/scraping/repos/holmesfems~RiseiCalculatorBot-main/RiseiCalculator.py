import os, sys, io, re
import discord
from discord import app_commands,Interaction
from discord.app_commands import Choice
from discord.utils import MISSING
from discord.ext import tasks
import traceback
from recruitment import recruitment,recruitFromOCR
import happybirthday.happybirthday as birthday
from openaichat.assistantChat import ChatSessionManager as chatbot
from riseicalculator2.riseicalculatorprocess import CalculatorManager,CalculateMode,getStageCategoryDict,DEFAULT_CACHE_TIME,DEFAULT_SHOW_MIN_TIMES
from typing import List,Dict,Literal
import datetime
from charmaterials.charmaterials import OperatorCostsCalculator
from rcutils import sendReplyToDiscord
from rcutils.rcReply import RCReply

TOKEN = os.environ["BOT_TOKEN"]
ID = os.environ["BOT_ID"]
url_botCommands = f"https://discord.com/api/v8/applications/{ID}/commands"
intents=discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents,command_prefix = '/')
t_delta = datetime.timedelta(hours=9)  # 9時間
JST = datetime.timezone(t_delta, 'JST')  # UTCから9時間差の「JST」タイムゾーン

def showException():
    ex_type, ex_value, ex_traceback = sys.exc_info()
    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)

    # Format stacktrace
    stack_trace = list()

    for trace in trace_back:
        stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s\n" % (trace[0], trace[1], trace[2], trace[3]))
    msg = "想定外のエラー:\n"
    msg += "Exception type : %s \n" % ex_type.__name__
    msg += "Exception message : %s\n" % ex_value
    msg += "Stack trace:\n"
    for item in stack_trace:
        msg += item
    return msg

def safeCallChoiceVal(choice):
    if choice is None:
        return None
    if(isinstance(choice,Choice)):
        return choice.value
    return choice
    
tree = app_commands.CommandTree(client)

async def riseicalculatorMaster(inter:Interaction,target:str,target_item:str=None,
                          event_code:str = None, mode:Literal["sanity","time"] = "sanity",min_times:int=DEFAULT_SHOW_MIN_TIMES,min_basetimes:int=3000,max_items:int=15,csv_file:bool = False,is_global:bool=True,cache_time:int = DEFAULT_CACHE_TIME):
    msg = ""
    try:
        mode = CalculateMode(mode)
        await inter.response.defer(thinking=True)
        msg = CalculatorManager.riseicalculatorMaster(target,target_item,event_code,is_global,mode,min_basetimes,cache_time,min_times,max_items,csv_file)
        await sendReplyToDiscord.followupToDiscord(inter,msg)

    except Exception as e:
        msg = showException()
    finally:
        print(msg)
        #channel = inter.channel()

targetItemChoice=[Choice(name=v["to_ja"],value=x) for x,v in getStageCategoryDict(False).items()]
modeChoice = [Choice(name="Sanity",value ="sanity"),Choice(name="Time",value ="time")]

@tree.command(
    name = "riseicalculator",
    description = "理性価値表計算,設定項目が複雑なので上級者向け。代わりに/riseimetarials,/riseistages,/riseievents,/riseilistsを使ってください",
)
@app_commands.describe(
    target = "どの項目を計算してほしい？",
    target_item = "検索したい素材名",
    event_code = "マップ名の中に含まれる文字列",
    mode = "計算モード選択",
    min_times = "計算に必要な最小サンプル数",
    min_basetimes = "基準マップとして選ばれるために必要な最小サンプル数",
    max_items = "表示するマップの数",
    csv_file = "理性価値表CSVファイルを添付する",
    is_global = "True:グローバル版基準の計算、False:大陸版の新ステージと新素材を入れた計算",
    cache_time = "計算キャッシュを保持する時間(分)"
)
@app_commands.choices(
    target = [
        Choice(name = "基準マップ", value = "basemaps"),
        Choice(name = "理性価値表", value = "san_value_lists"),
        Choice(name = "昇進素材別検索(target_item指定)",value = "items"),
        Choice(name = "通常ステージ検索(event_code指定)",value = "zone"),
        Choice(name = "イベント検索(event_code指定)",value = "events"),
        Choice(name = "初級資格証効率表",value = "te2list"),
        Choice(name = "上級資格証効率表",value = "te3list"),
        Choice(name = "特別引換証効率表",value = "special_list"),
        Choice(name = f"契約賞金引換効率表(CC#{CalculatorManager.CC_NUMBER})",value = "cclist"),
    ],
    target_item = targetItemChoice,
    mode = modeChoice
)
async def riseicalculator(inter:Interaction,target:Choice[str],target_item:Choice[str]=None,
                          event_code:str = None, mode:Choice[str]="sanity",min_times:int=DEFAULT_SHOW_MIN_TIMES,min_basetimes:int=3000,max_items:int=15,csv_file:bool = False,is_global:bool=True,cache_time:int = DEFAULT_CACHE_TIME):
    target = safeCallChoiceVal(target)
    target_item = safeCallChoiceVal(target_item)
    mode = safeCallChoiceVal(mode)
    await riseicalculatorMaster(inter,target,target_item,event_code,mode,min_times,min_basetimes,max_items,csv_file,is_global,cache_time)

    #print(rc.convert_rules)

@tree.command(
    name="riseimaterials",
    description="昇進素材の効率の良い恒常ステージを調べます。"
)
@app_commands.describe(
    target_item = "昇進素材を選択",
    mode = "計算モード選択",
    is_global = "True:グローバル版基準の計算(デフォルト)、False:大陸版の新ステージと新素材を入れた計算",
    csv_file = "ステージドロップ率をExcelとして出力する"
)
@app_commands.choices(
    target_item = targetItemChoice,
    mode = modeChoice
)
async def riseimaterials(inter:Interaction,target_item:Choice[str],mode:Choice[str]="sanity",is_global:bool=True,csv_file:bool=False):
    target_item = safeCallChoiceVal(target_item)
    mode = safeCallChoiceVal(mode)
    mode = CalculateMode(mode)
    await inter.response.defer(thinking=True)
    reply = CalculatorManager.riseimaterials(target_item,is_global,mode,toCsv=csv_file)
    await sendReplyToDiscord.followupToDiscord(inter,reply)

@tree.command(
    name="riseistages",
    description="恒常ステージの理性効率を検索します。恒常サイドストーリーも対象。"
)
@app_commands.describe(
    stage = "ステージ名を入力(例:1-7 SV-8 など)",
    mode = "計算モード選択",
    is_global = "True:グローバル版基準の計算(デフォルト)、False:大陸版の新ステージと新素材を入れた計算",
    csv_file = "ステージドロップ率をExcelとして出力する"
)
@app_commands.choices(
    mode = modeChoice
)
async def riseistages(inter:Interaction,stage:str,mode:Choice[str]="sanity",is_global:bool=True,csv_file:bool=False):
    _mode = safeCallChoiceVal(mode)
    stage = safeCallChoiceVal(stage)
    mode = CalculateMode(_mode)
    await inter.response.defer(thinking=True)
    reply = CalculatorManager.riseistages(stage,is_global,mode,toCsv=csv_file)
    await sendReplyToDiscord.followupToDiscord(inter,reply)

@riseistages.autocomplete("stage")
async def mainstage_autocomplete(inter:Interaction,current:str)->List[app_commands.Choice[str]]:
    strList = CalculatorManager.calculatorForMainland.autoCompleteMainStage(current)
    return [app_commands.Choice(name = name, value = value) for (name,value) in strList]

@tree.command(
    name="riseievents",
    description="期間限定イベントの理性効率を検索します。過去の開催済みイベントや、将来の未開催イベントも対象。"
)
@app_commands.describe(
    stage = "ステージ名を入力(例:SV-8 IW-8など)",
    mode = "計算モード選択",
    is_global = "True:グローバル版基準の計算(デフォルト)、False:大陸版の新ステージと新素材を入れた計算",
    csv_file = "ステージドロップ率をExcelとして出力する"
)
@app_commands.choices(
    mode = modeChoice
)
async def riseievents(inter:Interaction,stage:str,mode:Choice[str]="sanity",is_global:bool=True,csv_file:bool=False):
    _mode = safeCallChoiceVal(mode)
    stage = safeCallChoiceVal(stage)
    mode = CalculateMode(_mode)
    await inter.response.defer(thinking=True)
    reply = CalculatorManager.riseievents(stage,is_global,mode,toCsv=csv_file)
    await sendReplyToDiscord.followupToDiscord(inter,reply)

@riseievents.autocomplete("stage")
async def eventstage_autocomplete(inter:Interaction,current:str)->List[app_commands.Choice[str]]:
    strList = CalculatorManager.calculatorForMainland.autoCompleteEventStage(current)
    return [app_commands.Choice(name = name, value = value) for (name,value) in strList]

@tree.command(
    name="riseilists",
    description="理性効率表を出力します。"
)
@app_commands.describe(
    target = "表示する効率表を選んでください",
    mode = "計算モード選択",
    is_global = "True:グローバル版基準の計算(デフォルト)、False:大陸版の新ステージと新素材を入れた計算",
    csv_file = "理性価値表CSVファイルを添付する"
)
@app_commands.choices(
    target = [
        Choice(name = "基準マップ", value = "basemaps"),
        Choice(name = "理性価値表", value = "san_value_lists"),
        Choice(name = "初級資格証効率表",value = "te2list"),
        Choice(name = "上級資格証効率表",value = "te3list"),
        Choice(name = "特別引換証効率表",value = "special_list"),
        Choice(name = f"契約賞金引換効率表(CC#{CalculatorManager.CC_NUMBER})",value = "cclist"),
    ],
    mode = modeChoice
)
async def riseilists(inter:Interaction,target:Choice[str],mode:Choice[str]="sanity",is_global:bool=True,csv_file:bool=False):
    _mode = safeCallChoiceVal(mode)
    _target = safeCallChoiceVal(target)
    mode = CalculateMode(_mode)
    target = CalculatorManager.ToPrint(_target)
    await inter.response.defer(thinking=True)
    reply = CalculatorManager.riseilists(target,is_global,mode,toCsv=csv_file)
    await sendReplyToDiscord.followupToDiscord(inter,reply)

@tree.command(
    name="riseikakin",
    description="課金理性効率表を出力します。"
)
@app_commands.describe(
    target = "表示する効率表を選んでください",
    csv_file = "課金効率CSVファイルを添付する"
)
async def riseikakin(inter:Interaction,target:str,csv_file:bool = False):
    target = safeCallChoiceVal(target)
    csv_file = safeCallChoiceVal(csv_file)
    await inter.response.defer(thinking=True)
    reply = CalculatorManager.riseikakin(target,toCsv=csv_file)
    await sendReplyToDiscord.followupToDiscord(inter,msg=reply)

@riseikakin.autocomplete("target")
async def riseikakin_autoCompleteName(inter:Interaction,current:str)->List[app_commands.Choice[str]]:
    return [app_commands.Choice(name = name, value = value) for (name,value) in CalculatorManager.autoCompletion_riseikakin(current)]

#毎日3時に情報自動更新
@tasks.loop(time=datetime.time(hour=3, minute = 0, tzinfo=JST))
async def updateRiseiCalculatorInstances():
    CalculatorManager.updateAllCalculators()

class RecruitView(discord.ui.View):
    def __init__(self,timeout=180):
        super().__init__(timeout=timeout)
        self.jobAndPositionTags = []
        self.eliteTags = []
        self.otherTags = []
    
    @discord.ui.select(
        cls=discord.ui.Select,
        placeholder="職&位置タグ選択",
        options=[discord.SelectOption(label = x) for x in recruitment.jobTags + recruitment.positionTags],
        min_values=0,max_values=5
    )
    async def jobAndPosition_selected(self,inter:Interaction,select:discord.ui.Select):
        self.jobAndPositionTags = select.values
        await inter.response.defer()
    
    @discord.ui.select(
        cls=discord.ui.Select,
        placeholder="エリートタグ選択",
        options=[discord.SelectOption(label = x) for x in recruitment.eliteTags],
        min_values=0,max_values=2
    )
    async def elite_selected(self,inter:Interaction,select:discord.ui.Select):
        self.eliteTags = select.values
        await inter.response.defer()

    @discord.ui.select(
        cls=discord.ui.Select,
        placeholder="その他タグ選択",
        options=[discord.SelectOption(label = x) for x in recruitment.otherTags],
        min_values=0,max_values=5
    )
    async def other_selected(self,inter:Interaction,select:discord.ui.Select):
        self.otherTags = select.values
        await inter.response.defer()
    
    @discord.ui.button(
        label="★4確定のみ",style=discord.ButtonStyle.primary
    )
    async def excecuteHighRare(self,inter:Interaction,button:discord.ui.Button):
        await self.execute(inter,button,4)

    @discord.ui.button(
        label="すべて表示",style=discord.ButtonStyle.secondary
    )
    async def excecuteAll(self,inter:Interaction,button:discord.ui.Button):
        await self.execute(inter,button,1)
    
    async def execute(self,inter:Interaction,button:discord.ui.Button,minstar:int):
        selectedList = self.jobAndPositionTags+self.eliteTags+self.otherTags
        if(selectedList):
            await inter.response.defer(thinking=True)
            msg = recruitment.recruitDoProcess(selectedList,minstar)
            await sendReplyToDiscord.followupToDiscord(inter,msg)
        else:
            await inter.response.defer()

#recruitcal = app_commands.CommandTree(client)
@tree.command(
    name = "recruitsim",
    description = "公開求人検索 UI画面が出るのでそのままお使いください",
)
async def recruitsim(inter:Interaction):
    await inter.response.send_message(view=RecruitView(),ephemeral=True,delete_after=180.0)
    return

@tree.command(
    name = "recruitlist",
    description = "アークナイツ公開求人の高レア確定タグをすべて表示"
)
@app_commands.describe(
    star = "星の数",
    is_global = "True:グローバル版, False:大陸版"
)
@app_commands.choices(
    star = [Choice(name="4",value=4), Choice(name="5",value=5)]
)
async def recruitlist(inter:Interaction, star:Choice[int],is_global:bool = True):
    _star = safeCallChoiceVal(star)
    is_global = safeCallChoiceVal(is_global)
    await inter.response.defer(thinking=True)
    msg = recruitment.showHighStars(_star,is_global)
    await sendReplyToDiscord.followupToDiscord(inter,msg)

@tree.command(
    name = "operatormastercost",
    description= "オペレーターのスキル特化消費素材を調べる"
)
@app_commands.describe(
    operator_name = "オペレーターの名前、大陸先行オペレーターも日本語を入れてください",
    skill_num = "何番目のスキル",
)
@app_commands.choices(
    skill_num = [Choice(name=str(i),value=i) for i in range(1,4)],
)
async def operatormastercost(inter:Interaction,operator_name:str,skill_num:Choice[int]):
    operator_name = safeCallChoiceVal(operator_name)
    skill_num = safeCallChoiceVal(skill_num)
    await inter.response.defer(thinking=True)
    msg = OperatorCostsCalculator.skillMasterCost(operator_name,skill_num)
    await sendReplyToDiscord.followupToDiscord(inter,msg)

@operatormastercost.autocomplete("operator_name")
async def operator_name_autocomplete(inter:Interaction,current:str)->List[app_commands.Choice[str]]:
    strList = OperatorCostsCalculator.autoCompleteForMasterCost(current)
    return [app_commands.Choice(name = name, value = value) for name,value in strList]

@tree.command(
    name = "operatorelitecost",
    description= "オペレーターの昇進消費素材を調べる"
)
@app_commands.describe(
    operator_name = "オペレーターの名前、大陸先行オペレーターも日本語を入れてください",
)
async def operatorelitecost(inter:Interaction,operator_name:str):
    operator_name = safeCallChoiceVal(operator_name)
    await inter.response.defer(thinking=True)
    msg = OperatorCostsCalculator.operatorEliteCost(operator_name)
    await sendReplyToDiscord.followupToDiscord(inter,msg)
@operatorelitecost.autocomplete("operator_name")
async def operator_name_autocomplete_forelite(inter:Interaction,current:str)->List[app_commands.Choice[str]]:
    strList = OperatorCostsCalculator.autoCompleteForEliteCost(current)
    return [app_commands.Choice(name = name, value = value) for name,value in strList]

@tree.command(
    name = "operatormodulecost",
    description= "オペレーターのモジュール消費素材を調べる"
)
@app_commands.describe(
    operator_name = "オペレーターの名前、大陸先行オペレーターも日本語を入れてください",
)
async def operatormodulecost(inter:Interaction,operator_name:str):
    operator_name = safeCallChoiceVal(operator_name)
    await inter.response.defer(thinking=True)
    msg = OperatorCostsCalculator.operatorModuleCost(operator_name)
    await sendReplyToDiscord.followupToDiscord(inter,msg)
@operatormodulecost.autocomplete("operator_name")
async def operator_name_autocomplete_formodule(inter:Interaction,current:str)->List[app_commands.Choice[str]]:
    strList = OperatorCostsCalculator.autoCompleteForModuleCost(current)
    return [app_commands.Choice(name = name, value = value) for name,value in strList]

@tree.command(
    name="operatorcostlist",
    description="オペレーター消費素材の、いくつか役立つリストを出力します。"
)
@app_commands.describe(
    selection = "表示するリスト選択"
)
@app_commands.choices(
    selection = [
        Choice(name="星5昇進素材価値表",value="star5elite"),
        Choice(name="星6昇進素材価値表",value="star6elite"),
        Choice(name="未実装オペレーターの消費素材合計",value = "costofcnonly"),
        Choice(name="実装済オペレーターの消費素材合計",value = "costofglobal")
    ]
)
async def operatorcostlist(inter:Interaction,selection:Choice[str]):
    selection = safeCallChoiceVal(selection)
    selection = OperatorCostsCalculator.CostListSelection(selection)
    await inter.response.defer(thinking=True)
    msg = OperatorCostsCalculator.operatorCostList(selection)
    await sendReplyToDiscord.followupToDiscord(inter,msg)

CHANNEL_ID_HAPPYBIRTHDAY = int(os.environ["CHANNEL_ID_HAPPYBIRTHDAY"])

@tasks.loop(time=datetime.time(hour=0, minute=0, tzinfo=JST))
async def checkBirtyday():
    if(not CHANNEL_ID_HAPPYBIRTHDAY): return
    now=datetime.datetime.now(tz=JST)
    msg = birthday.checkBirthday(now)
    if(msg is not None):
        channel = client.get_channel(CHANNEL_ID_HAPPYBIRTHDAY)
        await sendReplyToDiscord.sendToDiscord(channel,msg)

MEMBERGUILD = int(os.environ["F_GUILDID"])
def checkIsMember(user:discord.User) -> bool:
    fserver = client.get_guild(MEMBERGUILD)
    YOUTUBEMEMBER_ROLE = int(os.environ["YOUTUBE_ROLEID"])
    youtubeMember = fserver.get_role(YOUTUBEMEMBER_ROLE)
    SERVERBOOSTER_ROLE = int(os.environ["BOOSTER_ROLEID"])
    serverBooster = fserver.get_role(SERVERBOOSTER_ROLE)
    def userIsInRole(user:discord.User,role:discord.Role):
        return user.id in [member.id for member in role.members]
    if userIsInRole(user,serverBooster): return True
    if userIsInRole(user,youtubeMember): return True
    return False

OPENAI_CHANNELID = int(os.environ["OPENAI_CHANNELID"])
async def msgForAIChat(message:discord.Message,threadName:str):
    messageText = message.content
    print(f"{messageText=}")
    async with message.channel.typing():
        chatReply = await chatbot.doChat(threadName,messageText,message.attachments)
        channel = message.channel
        files = [discord.File(io.BytesIO(file.bytesData),filename=file.filename) for file in chatReply.fileList]
        await channel.send(content = chatReply.msg,files=files)
        for item in chatReply.rcReplies:
            await sendReplyToDiscord.sendToDiscord(channel,item)

RECRUIT_CHANNELID = int(os.environ["RECRUIT_CHANNELID"])
async def msgForOCR(message:discord.Message):
    attachment = message.attachments
    if(not attachment): return
    for file in attachment:
        if(not file.width or not file.height): return
        image = file.url
        tagMatch = recruitFromOCR.taglistFromImage(image)
        print("タグを読みました",tagMatch)
        if(not tagMatch):return
        msg = recruitment.recruitDoProcess(tagMatch.matches,4,isGlobal=tagMatch.isGlobal)
        await sendReplyToDiscord.replyToDiscord(message,msg)
        if(tagMatch.isIllegal()):
            await sendReplyToDiscord.replyToDiscord(message,RCReply(
                plainText="タグが欠けているようね。上の計算結果に足りないタグを日本語でリプすれば、再計算させていただきますわ。詳しくはチャンネル概要見てね。\n(**このメッセージではなく、上の計算結果にリプしてね**)"
            ))

async def msgForOCRReply(message:discord.Message,referencedMessage:discord.Message):
    if(not (embeds := referencedMessage.embeds)):
        await sendReplyToDiscord.replyToDiscord(message,RCReply(
            plainText="返信メッセージが違うわ。計算結果の方にリプしてちょうだい。"
        ))
        return
    def splitBySpacestrings(string:str):
        return re.split(r"(?:\s|\n|　)+",string)
    addingCommands = splitBySpacestrings(message.content)
    if not addingCommands:
        return
    if (embedsTitle:= embeds[0].title) is None: return
    isGlobal = True
    mainlandMark = "(大陸版)"
    if(mainlandMark in embedsTitle):
        isGlobal = False
        embedsTitle = embedsTitle.replace(mainlandMark,"")
    existingTags = splitBySpacestrings(embedsTitle)
    resultTags = existingTags
    abbreviations = {
        "上エリ": "上級エリート",
        "エリ": "エリート",
        "COST": "COST回復",
        "コスト": "COST回復",
        "コスト回復": "COST回復",
    }
    def formatToTags(command:str):
        command.replace("タイプ","")
        return abbreviations.get(command,command)
    def isNullOrEmpty(tag:str):
        return not tag or tag.isspace()
    def isIlligal(tag:str):
        return tag not in recruitment.tagNameList
    def remove_blank_strings(string_list:List[str]):
        # Remove strings that are either empty or contain only whitespace
        return [string for string in string_list if string and not string.isspace()]
    #返信先のEmbedsのタイトルに問題があるとき
    if any(isIlligal(tag) for tag in existingTags):
        return
    for command in addingCommands:
        commandTags = re.split(r"(?:->)|→",command)
        commandTags = [formatToTags(tag) for tag in commandTags]
        #Check Illigal
        illigalTags = [tag for tag in commandTags if isIlligal(tag) and not isNullOrEmpty(tag)]
        if(illigalTags):
            await sendReplyToDiscord.replyToDiscord(message,RCReply(
                plainText=f"{illigalTags}のタグが違いますわ。もう一度入力してちょうだい。"
            ))
            return
        if(len(commandTags) == 1):
            #直接追加
            resultTags.append(commandTags[0])
        elif(len(commandTags) == 2):
            #書き換え
            old = commandTags[0]
            new = commandTags[1]
            resultTags = [new if item == old else item for item in resultTags]
    resultTags = remove_blank_strings(resultTags)
    resultTags = set(resultTags)
    if(len(resultTags) > recruitment.MAX_TAGCOUNT+2):
        await sendReplyToDiscord.replyToDiscord(message,RCReply(
            plainText=f"タグが多すぎるわ。5件ぐらいまでにしてちょうだい。"
        ))
        return
    msg = recruitment.recruitDoProcess(resultTags,4,isGlobal=isGlobal)
    await sendReplyToDiscord.replyToDiscord(message,msg)

async def msgForDM(message:discord.Message):
    if(not checkIsMember(message.author)):
        msg = "【自動返信】DMありがとうございます！\n"
        msg += "アステシアちゃんとお話をお楽しみいただくには、F鯖に加入の上、Youtubeアカウントと連携してふぉめの**Youtubeチャンネルメンバー登録**、もしくは**F鯖のサーバーブースト**をして頂く必要がございます！\n"
        msg += "ふぉめチャンネルはこちら: https://www.youtube.com//holmesfems\n"
        msg += "F鯖はこちら: https://discord.gg/arknightsflame\n"
        msg += "こちらの機能は有料限定であること、どうかご了承くださいまし:woman_bowing:"
        message.channel.send(msg)
    else:
        print("DM Recieved from: "+str(message.author))
        await msgForAIChat(message,str(message.author.id))
        
MAXMSGLEN = 200

@client.event
async def on_message(message:discord.Message):
    if(message.author.bot): return
    if message.channel.id == OPENAI_CHANNELID:
        await msgForAIChat(message,"public")
    elif message.channel.id == RECRUIT_CHANNELID:
        if message.reference is not None:
            referenced_message = await message.channel.fetch_message(message.reference.message_id)
            if(referenced_message.author != client.user):
                return
            await msgForOCRReply(message,referenced_message)
        await msgForOCR(message)
    elif message.channel.type is discord.ChannelType.private:
        await msgForDM(message)

@client.event
async def on_ready():
    await tree.sync()
    checkBirtyday.start()
    print('Botでログインしました')

client.run(TOKEN)