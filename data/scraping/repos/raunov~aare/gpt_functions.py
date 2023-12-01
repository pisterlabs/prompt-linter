import re
import os
from datetime import datetime
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())

# we need to store the available workers globally to be used in the chat_dispatcher and individual workers. 
# Each worker has a function to be called when the worker is selected.
#    * börsiuudiste_haldur: kasutaja küsimus on seotud uudistega konkreetse ettevõtte kohta. 
#    * aktsianalüütik: kasutaja küsimus on seotud sooviga analüüsida konkreetse ettevõtte aktsiat
#    * üldnõustaja: kasutaja soovib nõu mingi teema kohta
#    * vestluskaaslane: kasutaja ei soovi midagi eelpoolnimetatut, vaid lihtsalt vestelda
class Workers:
    def __init__(self):
        self.available_workers = {
            "börsiuudiste nõunik": "kasutaja küsimus on seotud uudistega konkreetse ettevõtte kohta.",
            "aktsianalüütik": "kasutaja küsimus on seotud sooviga analüüsida konkreetse ettevõtte aktsiat",
            "üldnõustaja": "kasutaja soovib nõu mingi teema kohta",
            "vestluskaaslane": "kasutaja ei soovi midagi eelpoolnimetatut, vaid lihtsalt vestelda"
        }

    def get_worker(self, worker_key):
        return self.available_workers.get(worker_key)

    def get_all_workers(self):
        return self.available_workers
# Create an instance of the Workers class
workers_instance = Workers()


def chat_dispatcher(user_input):
    # Create a ChatOpenAI instance with the specified model_name and temperature
    chat = PromptLayerChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # Convert the workers dictionary to a formatted string
    # Get all available workers
    all_workers = workers_instance.get_all_workers()
    workers_str = ""
    for key, description in all_workers.items():
        workers_str += f"**{key}**: {description}\n"

    template = """
    Sa oled dispetser, kes otsustab kasutaja päringu põhjal, mis tüüpi päringuga on tegemist ja suunab selle vastavale spetsialistile. 
    Vasta ainult spetsialisti nimega, ja ei midagi muud (kirjavahemärgid, tühikud jms). Sinu roll ei ole anda nõu, vaid otsustada ainult suunamise üle.
    Kui kasutaja päring ei ole seotud ühegi spetsialistiga, siis suuna see vestluskaaslasele. 
    Olemasolevad spetsialistid on:
    {workers_str}
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Siin on kasutaja päring: {user_input}. \nSpetsialist kelle poole suunata:"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, workers_str=workers_str)
    return response

def stocknews(user_input):
    from query_news import query_news
    # check the storage folder for the folders for companies that you have stocknews index data for. 
    # lets create a log file for the requests in the storage folder called "requests_log.txt"
    # inside the text file, write date and time of the request and the request itself.
    
    storage_folder = "storage"
    # list all folders in the storage folder
    companies = [f for f in os.listdir(storage_folder) if os.path.isdir(os.path.join(storage_folder, f))]
    # replace the underscore with a space in the folder names
    companies = [f.replace("_"," ") for f in companies]
    # add null to the list of companies
    companies.append("null")
    chat = PromptLayerChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    template = """
    Sa oled ettevõtte nimetuvastaja, kelle ülesanne on tuvastada, kas kasutaja küsimuses sisalduv ettevõte on sinu valikus [] märkide vahel. 
    Sinu roll ei ole anda nõu, ega vestelda, vaid ainult tuvastada, kas ettevõte on sinu valikus või mitte.
    Kui sul on kasutaja soovitud ettevõte nimi valikus olemas, siis vasta täpselt selle ettevõtte nimega, ei midagi rohkemat.
    Kui kui ettevõtte nime ei ole valikus, vasta: null
    Valikus olevad ettevõtted: {companies}. 
    Näiteks:
    Tesla
    null
    LHV Group
    NVIDIA
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    human_template = """
    Kasutaja küsimus: {user_input}
    Sinu vastus valikust: {companies}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, companies=companies)
    # if the response is not null, then the company is in the list of companies
    # we need to replace the spaces with underscores in the company name and remove any punctuation or trailing spaces
    if response != "null":
        company = response.replace(" ","_").strip(".,!?")
        # check if the company folder exists in the storage folder
        if os.path.isdir(os.path.join(storage_folder, company)):
            # data is there, query_news function will return the news
            response = query_news(user_input, company,"gpt-4")
        
    # log the request in the requests_log.txt uf8 file (date, time, company, user input)
    with open(os.path.join(storage_folder, "requests_log.txt"), "a", encoding="utf8") as f:
        f.write(f"{datetime.now()},'{company}','{user_input}'\n")
    return response


def kokkuvote(user_input, name="Aare"):

    # extract an url from user input
    url = re.search("(?P<url>https?://[^\s]+)", user_input).group("url")

    chat = PromptLayerChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
    template = """
    Sa oled abivalmis assistant.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = """
    Kasutaja soovib, et sa aitaksid teha kokkuvõtte dokumendist aadressilt {url}, aga sa veel ei oska seda, sest oled alles arendamisel. 
    Vasta viisakalt, et sa veel ei oska seda teha.
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, name=name)
    return response

def soovitus(user_input, name="Aare"):
    # Create a ChatOpenAI instance with the specified model_name and temperature
    chat = PromptLayerChatOpenAI(model_name="gpt-4", temperature=0.5)

    template = """
    Sa oled isiklik nõustaja, kes aitab leida lahendusi kliendi probleemidele või küsimustele. 
    Sinu eesmärk on aidata kasutajal kiiresti leida lahendus, mida ta otsib, ning seejärel veenvalt põhjendada oma soovitust.
    Ole konstruktiivne, lühike ja konkreetne, ära raiska aega tervitusteks, mine kohte asja juurde.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    human_template = """
    Aita kasutajal leida lahendus vastavalt päringule, mis on <päring></päring> märkide vahel. 
    Päring võib sisaldada probleemi või küsimust, millele kasutaja vastust otsib. 
    Ole konkreetne ja selge, struktureeri oma vastus loogiliselt. 
    Sõltumate küsimuse keelest, sina vastad eesti keeles.
    Kui sa ei tea vastust, siis ütle, et ei tea.
    
    Kasutaja päring:<päring>{user_input}</päring>
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, name=name)
    return response

def detect_stock_ticker(user_input):
    chat = PromptLayerChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    template = """
    You are a stock ticker detector, who helps to find or validates the stock tickers for the companies in the user input text.
    It can be from a major stock exchange such as NYSE or NASDAQ, or from a regional exchange. 
    Only if the company is from a regional exchange, add the exchange code to the ticker. You only output a ticker, nothing else. If you don't recognize any ticker, output null.
    regional exchanges supported:
    Tallinn Stock Exchange - .TL
    Helsinki Stock Exchange - .HE
    Riga Stock Exchange - .RG
    Vilnius Stock Exchange - .VS
    Stockholm Stock Exchange - .ST
    London Stock Exchange - .L
    XETRA - .DE
    
    From Baltic exchanges these are available companies and their tickers:
    Apranga: APG1L.VS
    Arco Vara: ARC1T.TL
    AUGA group: AUG1L.VS
    Baltika: BLT1T.TL
    Coop Pank: CPA1T.TL
    DelfinGroup: DGR1R.RG
    Ekspress Grupp: EEG1T.TL
    EfTEN Real Estate Fund: EFT1T.TL
    Enefit Green: EGR1T.TL
    Grigeo: GRG1L.VS
    Harju Elekter Group: HAE1T.TL
    Hepsor: HPR1T.TL
    INDEXO: IDX1R.RG
    Ignitis grupė: IGN1L.VS
    Klaipėdos nafta: KNF1L.VS
    LHV Group: LHV1T.TL
    Linas Agro Group: LNA1L.VS
    Merko Ehitus: MRK1T.TL
    Nordecon: NCN1T.TL
    Novaturas: NTU1L.VS
    Pro Kapital Grupp: PKG1T.TL
    PRFoods: PRF1T.TL
    Panevėžio statybos trestas: PTR1L.VS
    Pieno žvaigždės: PZV1L.VS
    Rokiškio sūris: RSU1L.VS
    Šiaulių bankas: SAB1L.VS
    SAF Tehnika: SAF1R.RG
    Silvano Fashion Group: SFG1T.TL
    Tallink Grupp: TAL1T.TL
    Telia Lietuva: TEL1L.VS
    Tallinna Kaubamaja Grupp: TKM1T.TL
    Tallinna Sadam: TSM1T.TL
    Tallinna Vesi: TVE1T.TL
    Vilkyškių pieninė: VLP1L.VS

    Example inputs and corresponding results:
    
    
    Input: This is a Tesla competitor
    Result: TSLA
    
    Input: The Apple computers inc is a great company
    Result: AAPL
    
    Input: Mis firma see Mikrosoft oligi?
    Result: MSFT
    
    Input: Ma ei teagi kuidas Lhv aktsia kood on
    Result: LHV1T.TL
    
    Input: Was ist Deutche Telekom bekannt als?
    Result: DTE.DE
    
    Input: Transferwise on nüüd Wise
    Result: WISE.L
    
    Input: Vanaema vaaritas mulle suppi
    Result: null
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    human_template = """
    Input: {user_input}
    Result:
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input)
    return response


def analyze_stock(user_input, history, name="Aare"):
    import json
    from stock_data import modules, quotes
    # Create a ChatOpenAI instance with the specified model_name and temperature
    chat = PromptLayerChatOpenAI(model_name="gpt-4", temperature=0)
    
    history_txt = "\n".join([f"{user}: {message}" for user, message in history])
    
    ticker = detect_stock_ticker(user_input)
    context = json.dumps(quotes(ticker)) + "\n" + json.dumps(modules(ticker, "asset-profile,default-key-statistics"))

    
    # Define a template string for the system message prompt
    template = """
    Sa oled professionaalne investeerimisassistent {name}, kes analüüsib aktsiaid. 
    Sinu eesmärk on kasutades konteksti aidata kasutajal kiiresti analüüsida aktsiat ning alati veenvalt põhjendada oma soovitusi.
    Oled täpne, ja konkreetne, ära raiska aega tervitusteks, mine kohte asja juurde.
    
    Sõltumata küsimuse ja olemasoleva konteksti keelest, sina vastad eesti keeles.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    human_template = """
    Aita kasutajat analüüsida vastavalt päringule, mis on <päring></päring> märkide vahel. 
    Pead võtma arvesse ettevõtte spetsiifilist konteksti, mis asub <kontekst></kontekst> märkide vahel json formaadis.
    Kui on vaja, siis võta arvesse ka vestlusajalugu <vestlusajalugu></vestlusajalugu>.
    Ole konkreetne ja selge, struktureeri oma vastus loogiliselt.
    Sõltumata küsimuse ja olemasoleva konteksti keelest, sina vastad eesti keeles.
    Ära mõtle ise infot välja, kui sa ei tea vastust, siis ütle, et ei tea.
    
    Vestlusajalugu:<vestlusajalugu>
    {history_txt}
    </vestlusajalugu>
    
    Kontekts:<kontekst>
    {context}
    </kontekst>
    
    Kasutaja päring:
    <päring>{user_input}</päring>
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, name=name, history_txt=history_txt, context=context)
    
    return response

# history is in the following format:
# [('RaunoV', '@Aare millest me rääkisime?'), ('Aare', 'Ei mäleta')]
def chitchat(user_input, history, name="Aare"):

    # Get all available workers
    all_workers = workers_instance.get_all_workers()

    workers_str = ""
    for key, description in all_workers.items():
        workers_str += f"* {key}: {description}\n"
    
    # Create a ChatOpenAI instance with the specified model_name and temperature
    chat = PromptLayerChatOpenAI(model_name="gpt-4", temperature=0.7)

    # Define a template string for the system message prompt
    template = """
    Sa oled investeerimisteadlik vestluspartner nimega {name}, kes oskab suhelda erinevate inimestega, kohanedes nende vestlusstiiliga.
    Kui vestluspartner on lõbus, siis sina oled lõbus, kui vestluspartner on tõsine, siis sina oled tõsine.
    Kui vestlusajaloost selgub, et oled juba tervitanud vestluspartnerit, siis ära seda rohkem tee, vaid mine kohe teema juurde.
    Vestlusajaloos võivad olla erinevad vestluspartnerid, kellele saad vastata erinevalt. 
    Ole tähelepanelik, et sa ei vastaks kogemata valele vestluspartnerile. 
    Võta arvesse, et sinu enda nimi on {name}, seega ära korralda vestlust iseendaga, ja ära võta enda omaks teiste vestluspartnerite sõnumeid.
    Kui vestluspartner küsib, mida sa teed või oskad, siis tee kokkuvõte olemasolevate spetsialistide oskustest mina vormis (nagu sa oleksid kõik spetsialistid ühes isikus).
    Olemasolevad spetsialistid on:
    {workers_str}
    
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    # convert the history to a string in the format: User: message\nUser: message\nUser: message
    history_txt = "\n".join([f"{user}: {message}" for user, message in history])
    # list also all unique users in the history as comma separated string
    users = ", ".join(set([user for user, message in history]))
    
    human_template = """
    Viimane vestluspartneri sõnum, millele vastad on: 
    ----------------------------------------
    {user_input}
    ----------------------------------------
    
    Ole tähelepanelik, et sa ei vastaks kogemata valele vestluspartnerile. Vestlusajaloos vestluspartnerid kelle nimed on (sealhulgas sina: {name}):
    ----------------------------------------
    {users}
    ----------------------------------------
    
    Oma vastustes lähtud ka eelnevast vestlusest <eelnev_vestlus></eelnev_vestlus>, et olla võimalikult loomulik ning pakud vestlusele võimalikult palju väärtust. 
    Võid vastata ka teistele vestluspartneritele, kui see on vestluse jaoks vajalik ja loogiline.
    Kui sul ei ole vestlusse midagi lisada, siis ütle, et ei tea.
    <eelnev_vestlus>
    {history_txt}
    </eelnev_vestlus>
    
    SINU VASTUS:
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, history_txt=history_txt, name=name,users=users, workers_str=workers_str)
    
    return response

def chitchat_langchain():
    # we're going to use langchain and the ConversationSummaryBufferMemory for remembering the conversation.
    # It keeps buffer of recent interactions in memory, and compiles the old interactions into a summary and uses both.
    
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain.llms import PromptLayerOpenAIChat
    
    prompt="""
    Kokkuvõtke järk-järgult antud vestlus, lisades eelnevale kokkuvõttele, et saada uus kokkuvõte.

NÄIDE
Praegune kokkuvõte:
Inimene pärib AI-lt tehisintellekti kohta. AI peab tehisintellekti heaks jõuks.

Uued vestluse read:
Inimene: Miks pead sa tehisintellekti heaks jõuks?
AI: Tehisintellekt aitab inimesi nende täieliku potentsiaali saavutamisel.

Uus kokkuvõte:
Inimene pärib AI-lt tehisintellekti kohta. AI leiab, et tehisintellekt on hea jõud, sest aitab inimestel saavutada oma täielikku potentsiaali.
NÄITE LÕPP

Praegune kokkuvõte:
{summary}

Uued vestluse read:
{new_lines}

Uus kokkuvõte:
    """
    
    PromptTemplate = ChatPromptTemplate.from_template(prompt)
    llm=PromptLayerOpenAIChat(model_name="gpt-4", temperature=0.7)
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=650,prompt=PromptTemplate)
    memory.save_context({"input": "Tere, kuidas läheb?"},{"output":"Hästi, sul?"})
    memory.save_context({"input": "Hästi, sul?"},{"output":"Hästi, aitäh!"})
    memory.load_memory_variables({})
    messages = memory.chat_memory.messages
    previous_summary = ""
    response = memory.predict_new_summary(messages, previous_summary)
    
    return response