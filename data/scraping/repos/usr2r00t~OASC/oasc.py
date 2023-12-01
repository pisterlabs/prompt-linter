"""-------------------------------------------------------------------
# OASC - OpenAI Security Console
-------------------------------------------------------------------"""
__author__ = "z0nd3rl1ng" + "0xAsFi"
__version__ = "0.0.1"

"""----------------------------------------------------------------"""

# MODULE REQUIREMENT CHECK
try:
    import random, os, json, hashlib, time
    import openai, requests
    import pandas as pd
    from bs4 import BeautifulSoup as bs
    from web3 import Web3
    import stem
    import stem.connection
    import stem.process
    from requests.structures import CaseInsensitiveDict
    from censys.search import CensysHosts
except ModuleNotFoundError:
    print("[*] searching required modules ...")
    os.system("pip3 install requests")
    os.system("pip3 install openai")
    os.system("pip3 install beautifulsoup4")
    os.system("pip3 install lxml")
    os.system("pip3 install pandas")
    os.system("pip3 install web3")
    os.system("pip3 install censys")
    os.system("pip3 install stem")
    os.system("pip3 install hashlib")
    import random, os, json, hashlib, time
    import openai, requests
    import pandas as pd
    from bs4 import BeautifulSoup as bs
    from web3 import Web3
    from censys.search import CensysHosts
    import stem
    import stem.connection
    import stem.process
    from requests.structures import CaseInsensitiveDict
"""----------------------------------------------------------------"""

# GLOBAL VARIABLES
openai.api_key = "[ENTER YOUR API KEY HERE]"
numlookupapikey = "[ENTER YOUR API KEY HERE]"
cenapikey = "[ENTER YOUR API ID HERE]"
censecret = "[ENTER YOUR API SECRET HERE]"
virustotalapikey = "[ENTER YOUR API SECRET HERE]"
wigleapienc = "[ENTER YOUR ENCODED API KEY HERE]"
# OPENAI ENGINE AND FINETUNE PARAMETERS
ENGINE = "text-davinci-003"
TEMPERATURE = 0
MAX_TOKENS = 2048
# THIRD PARTY TOOLS
sherlock = "/home/z0nd3rl1ng/Tools/sherlock/sherlock"
exiftool = "exiftool"
torghost = "/home/z0nd3rl1ng/Tools/torghost.py"
"""----------------------------------------------------------------"""


# FUNCTION TO EXPORT ENVIRONMENT VARIABLES
def setEnvKeys():
    openaitoken = input("[OpenAI API Key]╼> ")
    os.system("export OPENAI_API_KEY='"+openaitoken+"'")
    numlookuptoken = input("[Numlookup API Key]╼> ")
    os.system("export NUMLOOKUP_API_KEY='" + numlookuptoken + "'")
    cenapitoken = input("[CenSys API Key]╼> ")
    os.system("export CENSYS_API_KEY='" + cenapitoken + "'")
    censecrettoken = input("[CenSys Secret Key]╼> ")
    os.system("export CENSYS_SECRET_KEY='" + censecrettoken + "'")


# FUNCTION TO SET FINETUNING FOR OPENAI REQUEST
def openaiFinetuning(engine, temperature, max_tokens):
    ENGINE = engine
    TEMPERATURE = temperature
    MAX_TOKENS = max_tokens


# FUNCTION TO LIST OPENAI ENGINES
def openaiEngines():
    engines = openai.Engine.list()
    for ids in engines.data:
        print(ids.id)


# FUNCTION TO EXPORT CONTENT TO FILE
def exportContent(data, path):
    with open(path, "w") as file:
        file.write(str(bs(data)))


# FUNCTION TO IMPORT CONTENT FROM FILE
def importContent(path):
    with open(path, "r") as file:
        content = file.readlines()
    content = "".join(content)
    prettyprompt = bs(content, "lxml")
    return prettyprompt


# FUNCTION FOR TOR NETWORK REQUEST
def torRequest(onionurl,path):
    def proxySession():
        # SET TOR AS PROXY
        session = requests.session()
        session.proxies = {'http':  'socks5://127.0.0.1:9050', 'https': 'socks5://127.0.0.1:9050'}
        return session

    tor_process = stem.process.launch_tor_with_config(config={'SocksPort': str(9050), 'ControlPort': str(9051)})
    try:
        request = proxySession()
        response = request.get(onionurl)
        exportContent(response, path)
    finally:
        tor_process.kill()


# FUNCTION FOR A BLOCKCHAIN REQUEST
def blockchainRequest(network, address):
    if network == "1":
        blockchain = 'https://blockchain.info/rawaddr/' + address
        wallet = pd.read_json(blockchain, lines=True)
        balance = float(wallet.final_balance) / 100000000
        inbound = float(wallet.total_received) / 100000000
        outbound = float(wallet.total_sent) / 100000000
        print("\n[*] BALANCE:\t" + str(balance) + " BTC")
        print("[*] RECEIVED:\t" + str(inbound) + " BTC")
        print("[*] SENT:\t" + str(outbound) + " BTC\n")
    elif network == "2":
        blockchain = 'https://mainnet.infura.io/v3/64e9df670efb49ac9b71f9984f29dccd'
        web3 = Web3(Web3.HTTPProvider(blockchain))
        if web3.isConnected():
            balance = web3.eth.getBalance(address)
            print(web3.fromWei(balance, "ETH"))
    else:
        print(network+" is not supported yet!")


# FUNCTION FOR OPENAI REQUEST
def openaiRequest(type, interact):
    if type == "console":
        response = openai.Completion.create(
            engine=ENGINE,
            prompt=(f"{interact}"),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stop=None
        )
        response = response["choices"][0]["text"]
        return response


# FUNCTION FOR A CENSYS API REQUEST
def censysRequest(query):
    censyshost = CensysHosts(cenapikey,censecret)
    results = censyshost.search(query, per_page=5, pages=2)
    rs = results.view_all()
    hosts = censyshost.search(query, per_page=5, virtual_hosts="ONLY")
    hs = hosts()
    export = str(rs)+str(hs)
    exportContent(export, "report-"+query)


# FUNCTION TO GENERATE AI IMAGE WITH OPENAI
def openaiImageCreator(interact):
    response = openai.Image.create(prompt=interact, n=1, size="1024x1024")
    print("\n"+response['data'][0]['url'])


# FUNCTION TO ANALYZE FILE CONTENT
def openaiFileAnalyzer():
    path = input("[File Path]╼> ")
    content = importContent(path)
    prompt = "Describe following file content: " + str(content)
    type = "console"
    response = openaiRequest(type, prompt)
    print(response)


# FUNCTION TO CREATE FILE TEMPLATE
def openaiFileCreator():
    data = input("[Describe Content]╼> ")
    path = input("[File Path]╼> ")
    type = "console"
    response = openaiRequest(type, data)
    exportContent(response, path)


# FUNCTION FOR NUMLOOKUP API REQUEST
def numlookupRequest(mobilenumber):
    url = "https://api.numlookupapi.com/v1/validate/"+mobilenumber
    headers = CaseInsensitiveDict()
    headers["apikey"] = numlookupapikey
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        country_code = data["country_code"]
        carrier = data["carrier"]
        line_type = data["line_type"]
        country = data["country_name"]
        print("\nMobile Number:\t", mobilenumber)
        print("Country Code:\t", country_code)
        print("Carrier:\t", carrier)
        print("Line Type:\t", line_type)
        print("Country:\t", country)
    else:
        print("\nError retrieving data for mobile number:", mobilenumber)


# FUNCTION TO LIST SOCIAL AND REVERSE ENGINEERING MENU
def file():
    banner()
    print("\nFILE MENU\n")
    print("(1)Analyze File Content ")
    print("(2)Generate File Template")
    print("(3)Generate Image")
    print("(4)Dump Onion Site")
    print("(0)Back\n")
    mode = input("[Select Mode]╼> ")    
    if mode == "1":
        openaiFileAnalyzer()
    elif mode == "2":
        openaiFileCreator()
    elif mode == "3":
        interact = input("[Description]╼> ")
        openaiImageCreator(interact)
    elif mode == "4":
        onionurl = input("[Onion Url]╼> ")
        path = input("[File Path]╼> ")
        torRequest(onionurl,path)
    elif mode == "0":
        banner()
        openaiSecurityConsole()
    else:
        file()
        print("Wrong input, try again.")


# FUNCTION TO LIST OPSEC MENU
def opsec():
    banner()
    print("\nOPSEC MENU\n")
    print("(1)Redirect Traffic TorGhost")
    print("(2)Delete Meta Data")
    print("(3)Virustotal Scan")
    print("(0)Back\n")

    def startTorghost():
        os.system("sudo python3 "+torghost+" -s")
        bg = input("[Background(Y/n)]╼> ")
        if bg == "y":
            banner()
            openaiSecurityConsole()
        elif bg == "Y":
            banner()
            openaiSecurityConsole()
        else:
            stopTorghost()

    def stopTorghost():
        os.system("sudo python3 "+torghost+" -x")

    def deleteExif(folder):
        os.system(exiftool+" -all= "+folder)
    
    def virustotalScan(filepath):
        endpoint = 'https://www.virustotal.com/vtapi/v2/file/report'
        params = {'apikey': virustotalapikey, 'resource': hashlib.md5(open(filepath, 'rb').read()).hexdigest()}
        response = requests.get(endpoint, params=params)
        while response.json().get('response_code') == 0:
            print("running scan. report not ready yet, waiting 60 seconds...")
            time.sleep(60)
            response = requests.get(endpoint, params=params)
            print(response.json())
        return response.json()

    mode = input("[Select Mode]╼> ")
    if mode == "1":
        startTorghost()
    elif mode == "2":
        folder = input("[Path]╼> ")
        deleteExif(folder)
    elif mode == "3":
        filepath = input("[Path]╼> ")
        report = virustotalScan(filepath)
        print(report)
    elif mode == "0":
        banner()
        openaiSecurityConsole()
    else:
        print("Wrong input, try again.")
        opsec()


# FUNCTION TO LIST OSINT MENU
def osint():
    banner()
    print("\nOSINT MENU\n")
    print("(1)Host Reconnaissance")
    print("(2)People Reconnaissance")
    print("(3)Phone Number Lookup")
    print("(4)Crypto Wallet Tracker")
    print("(5)List Meta Data")
    print("(6)Access Point Tracker")
    print("(0)Back\n")

    def hostReconnaissance():
        print("\nScanning target with censys search\n")
        query = input("[Domain]╼> ")
        censysRequest(query)

    def peopleReconnaissance():
        print("\nPEOPLE RECONNAISSANCE\n")
        print("(1)General Search")
        print("(2)Username Search")
        print("(3)Name Search")
        print("(0)Back\n")

        def generalSearch(query):
            print("\nSearching information for "+query+"\n")
            os.system('open -a "Google Chrome" "https://www.google.com/search?q=allintext:'+query+'"')

        def usernameSearch(username):
            print("\nSearching for "+username+"\n")
            os.system('python3 '+sherlock+' '+username)

        def nameSearch(fullname):
            print("\nSearching information for "+fullname+"\n")
            print("\nFacebook:\n")
            os.system('open -a "Google Chrome" "https://www.google.com/search?q='+fullname+' site:facebook.com"')
            print("\nLinkedIn:\n")
            os.system('open -a "Google Chrome" "https://www.google.com/search?q='+fullname+' site:linkedin.com"')

        mode = input("[Select Mode]╼> ")
        if mode == "1":
            query = input("[Search Query]╼> ")
            generalSearch(query)
        elif mode == "2":
            username = input("[Username]╼> ")
            usernameSearch(username)
        elif mode == "3":
            fullname = input("[Full Name]╼> ")
            nameSearch(fullname)
        elif mode == "0":
            osint()
        else:
            print("Wrong input, try again.")
            peopleReconnaissance()

    def phoneNumber():
        print("\nSearching for phone number information.\n")
        mobilenumber = input("[Mobile Number]╼> ")
        numlookupRequest(mobilenumber)

    def coinHunter():
        print("\nCoin Hunter - Crypto Wallet Tracker\n")
        print("(1)Bitcoin Mainnet")
        print("(2)Ethereum Mainnet")
        print("(0)Back\n")
        network = input("[Select Network]╼> ")
        if network == "1":
            address = input("[Wallet Address]╼> ")
            blockchainRequest("1", address)
        elif network == "2":
            address = input("[Wallet Address]╼> ")
            blockchainRequest("2", address)
        elif network == "0":
            osint()
        else:
            print("Wrong input, try again.")
            coinHunter()

    def listExif(folder):
        os.system(exiftool +" "+folder)

    def apTracker():
        print("\nAccess-Point Tracker - WiGLE\n")
        print("(1)SSID")
        print("(2)BSSID/MAC")
        print("(3)LOCATION")
        print("(0)Back\n")
        mode = input("[Select Mode]╼> ")
        if mode ==  "1":
            ssid = input("[SSID]╼> ")
            url = f"https://api.wigle.net/api/v2/network/search?onlymine=false&ssid={ssid}"
            headers = {"Authorization": f"Basic {wigleapienc}"}
            response = requests.get(url, headers=headers)
            if response.ok:
                for entry in response.json()["results"]:
                    print(str(entry)+"\n")
                    print("https://www.google.de/maps/@"+str(entry["trilat"])+","+str(entry["trilong"])+",20z")
            else:
                print(response.raise_for_status())
        elif mode == "2":
            bssid = input("[BSSID]╼> ")
            url = f"https://api.wigle.net/api/v2/network/detail?netid={bssid}"
            headers = {"Authorization": f"Basic {wigleapienc}"}
            response = requests.get(url, headers=headers)
            if response.ok:
                for entry in response.json()["results"]:
                    print(str(entry)+"\n")
                    print("https://www.google.de/maps/@"+str(entry["trilat"])+","+str(entry["trilong"])+",20z")
            else:
                print(response.raise_for_status())
        elif mode == "3":
            latitude = input("[LATITUDE]╼> ")
            longitude = input("[LONGITUDE]╼> ")
            url = f"https://api.wigle.net/api/v2/network/search?onlymine=false&latrange1={latitude}&latrange2={latitude}&longrange1={longitude}&longrange2={longitude}"
            headers = {"Authorization": f"Basic {wigleapienc}"}
            response = requests.get(url, headers=headers)
            if response.ok:
                response.json()
            else:
                print(response.raise_for_status())
        else:
            osint()

    mode = input("[Select Mode]╼> ")
    if mode == "1":
        hostReconnaissance()
    elif mode == "2":
        peopleReconnaissance()
    elif mode == "3":
        phoneNumber()
    elif mode == "4":
        coinHunter()
    elif mode == "5":
        folder = input("[Path]╼> ")
        listExif(folder)
    elif mode == "6":
        apTracker()
    elif mode == "0":
        banner()
        openaiSecurityConsole()
    else:
        osint()
        print("Wrong input, try again.")


# FUNCTION TO LIST HELP MENU - COULD BE SWAGGED UP ;)
def help():
    print("\nCOMMANDS\tDESCRIPTION\n")
    print("help\t\tprint this help menu")
    print("clear\t\tclear screen / refresh banner")
    print("file\t\tai content analyzer and creator")
    print("osint\t\topen source intelligence")
    print("opsec\t\toperation security")
    print("exit\t\tquit oasc\n")
    print("other inputs interact directly with openAI\n")


# FUNCTION FOR THE OPENAI QUERY PROMPT (CORE-SYSTEM)
def openaiSecurityConsole():
    while True:
        interact = input("[OASC]╼> ")
        # SYSTEM COMMAND HANDLER 
        if interact == "exit":
            exit()
        elif interact == "file":
            file()
        elif interact == "osint":
            osint()
        elif interact == "opsec":
            opsec()
        elif interact == "help":
            help()
        elif interact == "clear":
            banner()
        else:
            type = "console"
            response = openaiRequest(type, interact)
            print(response)


# FUNCTION FOR A CALLABLE BANNER
def banner():
    os.system("clear")
    padding = '  '
    O = [[' ','┌','─','┐'],
	     [' ','│',' ','│'],
	     [' ','└','─','┘']]
    A = [[' ','┌','─','┐'],
	     [' ','├','─','┤'],
	     [' ','┴',' ','┴']]
    S = [[' ','┌','─','┐'],
	     [' ','└','─','┐'],
	     [' ','└','─','┘']]
    C = [[' ','┌','─','┐'],
	     [' ','│',' ',' '],
	     [' ','└','─','┘']]
	
    banner = [O,A,S,C]
    final = []
    print('\r')
    init_color = random.randint(10,40)
    txt_color = init_color
    cl = 0

    for charset in range(0, 3):
        for pos in range(0, len(banner)):
            for i in range(0, len(banner[pos][charset])):
                clr = f'\033[38;5;{txt_color}m'
                char = f'{clr}{banner[pos][charset][i]}'
                final.append(char)
                cl += 1
                txt_color = txt_color + 36 if cl <= 3 else txt_color
            cl = 0
            txt_color = init_color
        init_color += 31
        if charset < 2: final.append('\n   ')

    print(f"   {''.join(final)}")
    print(f'{padding}  by z0nd3rl1ng & \n\t 0xAsFi\n')


# MAIN FUNCTION (ENTRY-POINT)
if __name__ == "__main__":
    banner()
    help()
    openaiSecurityConsole()

