import logging
import datetime
import google_images_download   #importing the library
from PIL import Image
import os
import random
from threading import Event
from time import time
from datetime import timedelta
import pytz
from pprint import pprint
import openai
import urllib
import SettingEnvVar


from telegram.ext import Updater, CommandHandler, DictPersistence, MessageHandler, Filters, CallbackContext
from telegram import ChatAction, Update
import telegram
from telegram.ext.dispatcher import run_async

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

shipPostingID = '-1001160290532'
openai.api_key = os.getenv('OPEN_TOKEN')

ship_types = ["Amphibious warfare","Barque","Barquentine","Battlecruiser","Galleon","Galleass","Galliot","Karve","Knarr",
        "Containers","Lorcha","Liberty","Mistico","Penteconter","Steamship","Trabaccolo","Trireme","Yacht",
        "Bulk Carrier","canoe","Tankers","Special Purposes","Offshores","Cabin cruiser","Cruise","Cable ferry","Canoe",
        "Cape Islander","Captain's gig","Car-boat","Car float","Catamaran","Center console","Chundan vallam","Coble","Cog","Coracle","Cornish pilot gig",
        "Crash rescue","Deck","Dhow","ekranoplano","Dhoni","Dinghy","Dorna","Dory","Dragon","Drift","Drifter fishing","Drifter naval",
        "Dugout","Durham","lucky", "lucky", "swimming dog", "Al Rekayyat, LNG Tanker","SS Atlantic Causeway","SS Atlantic Conveyor","Axel Mærsk","MV Acavus","MV Adula",\
        "Akebono Maru","MV Alexia","Altmark","MV Amastra","Amoco Cadiz","MV Ancylus","USS Adhara (AK-71)","USS Albireo (AK-90)",\
        "USS Alderamin (AK-116)","USS Alkaid (AK-114)","USS Alkes (AK-110)","USS Allegan (AK-225)","USS Allioth (AK-109)","USS Alnitah (AK-127)",\
        "USS Aludra (AK-72)","SS American Victory","MS Antenor","USS Appanoose (AK-226)","USS Ara (AK-136)","Aranui 3","USS Arided (AK-73)",\
        "USS Arkab (AK-130)","SS Arthur M. Anderson","USS Ascella (AK-137)","MV Ascension","Astron","USS Azimech (AK-124)",\
        "HMCS Labrador (1954-1961; transferred to Coast Guard)","Harry DeWolf-class offshore patrol vessel","HMCS Harry DeWolf (2019- (planned); under construction)",\
        "CGS Northern Light (1876-1890; sold)","CGS Stanley (1888-1935; broken up)","CGS Earl Grey (1909-1914; sold to Russia)",\
        "CGS Mikula (1916; 1923-1937; ex-J.D. Hazen, ex-Mikula Seleaninovich; broken up)","CCGS Saurel (1929-1967; broken up)",\
        "Almagrundet","Amorina","Bras d'Or","USS A-1","USC&GS A. D. Bache","USS Aaron Ward","USS Abarenda","USS Abbot","HMS Abdiel",\
        "Abeona (ship)","HMS Abercrombie","Aberdeen (ship)","SS Abessinia","HMS Abigail","USS Ability","USS Abner Read","HMS Aboukir",\
        "HNLMS Abraham Crijnssen","USS Abraham Lincoln","HNLMS Abraham van der Hulst","HDMS Absalon","USS Absecon","BNS Abu Bakar","Japanese Abukuma",\
        "HMS Abundance","HMCS Acadia","HMS Acasta","USS Accentor","USS Accomac","HMS Achates","HMS Acheron","French Achille","HMS Achille","HMS Achilles",\
        "USS Achilles","USS Acme","USS Acoma","French Aconit","HMS Acorn","HMS Actaeon","Active (ship)","HMS Active","USCGC Active","USRC Active",\
        "USS Active","HMS Acute","HMS Adam & Eve","Adamant (ship)","HMS Adamant","USS Adams","HMS Adder","HMAS Adelaide","HMS Adelaide","USS Adirondack",\
        "SS Adler","French Admirable","Russian Admiral Gorshkov","Admiral Kingsmill (ship)","Russian Admiral Lazarev",\
        "Soviet cruiser Admiral Nakhimov","Russian Admiral Nakhimov","Russian Admiral Ushakov","USS Admiral","HMS Admiralty","Adolf Vinnen",\
        "Greek Adrias","Adriatic (ship)","USS Adroit","USS Advance","HMS Advantage","Adventure (ship)","HMS Adventure","HMS Advice Prize","HMS Advice",\
        "USS Advocate","HNoMS Aeger","SS Aenos","SS Aeolus","HMS Aeolus","USS Aeolus","HMS Aetna","Greek Aetos","USS Affray","SS Afoundria","HMS Africa",\
        "French Africain","French Africaine","HMS Africaine","African Queen (ship)","HMS Afridi","HMS Agamemnon","HMT Agate","USS Agawam",\
        "HMS Aggressor","USS Aggressor","USS Agile","Agincourt (ship)","HMS Agincourt","MV Agios Georgios","Italian Agostino Barbarigo","BAP Aguirre",\
        "HMS Aid","ST Aid","Greek Aigaion","French Aigle","HMS Aigle","HMAS Air Hope","Italian Airone","SS Aisne","Ajax (ship)","HMS Ajax",\
        "SS Ajax","USS Ajax","Akademik Karpinsky","Japanese Akagi","Japanese Akashi","Japanese destroyer Akatsuki","HMS Akbar",\
        "Japanese destroyer Akebono","Japanese destroyer Akigumo","Japanese Akitsushima","Japanese destroyer Akizuki","USS Alabama","Alacrity (ship)",\
        "HMS Alacrity","USS Alacrity","Brazilian Alagoas","USS Alameda","PNS Alamgir","HMS Alarm","USS Alarm","SS Alaska","USS Alaska","RMS Alaunia",\
        "HMS Albacore","USS Albacore","HMS Alban","HMS Albany","USS Albany","Italian Albatros","List ofs named Albatross","HMS Albatross",\
        "HMS Albemarle","USS Albemarle","List ofs named Albion","HMS Albion","USS Albuquerque","Spanish destroyer Alcalá Galiano","HMS Alcantara",\
        "HMS Alceste","HMS Alcide","Italian Alcione","HMS Alcmene","SS Alcoa Puritan","USS Alcor","SS Alcyone Fortune","HMS Aldborough",\
        "Italian Aldebaran","HMS Alderney","SS Aldershot","HMS Alecto","CS Alert","HMS Alert","USS Alert","Alexander (East Indiaman)","Alexander (ship)",\
        "USS Alexander Hamilton","Russian Alexander Nevsky","HMS Alexander","French Alexandre","USS Alexandria","HMS Alexandria",\
        "Alfred (East Indiaman)","HMS Alfred","Italian Alfredo Cappellini","French Algerien","HMS Algerine","French Algesiras",\
        "USS Algol","HMCS Algonquin","USS Algonquin","USS Algorma","BNS Ali Haider","SS Alice","USS Alice","PS Alice Dean","Italian Aliseo",\
        "USS Allegheny","USS Allen","HMS Alliance","USS Alliance","HMS Alligator","USS Alligator","USS Alloway","Almeria Lykes","Brazilian Almirante Barroso",\
        "ARA Almirante Brown","Chilean Almirante Condell","Spanish destroyer Almirante Ferrandiz","BAP Almirante Grau",\
        "Chilean Almirante Latorre","Chilean Almirante Lynch","Chilean Almirante Riveros","HMCS CD 21","HMCS CD 22",\
        "HMCS CD 23","HMCS CD 24","HMCS CD 25","HMCS CD 26","HMCS CD 27","HMCS CD 28","Baleine","HMCS Constance[5]","HMCS Curlew[5]",\
        "HMCS P.V. I (PV type)[6]","HMCS P.V. II (PV type)","HMCS P.V. III (PV type)","HMCS P.V. IV (PV type)","HMCS P.V. V (PV type)","HMCS P.V. VI (PV type)",\
        "HMCS P.V. VII (PV type)","HMCS TR 1 (Castle class TR series)[4]","HMCS TR 2 (Castle class)","HMCS TR 10 (Castle class)","HMCS TR 11 (Castle class)",\
        "HMCS TR 12 (Castle class)","HMCS Tuna","CGS Earl Grey","CGS Minto","CGS Stanley","Achille Lauro","Andrea Doria","Argonaut (early submarine)","Argus",\
        "Beagle","Bismarck","Bounty","Britannic","Carpathia","Charlotte Dundas","Clermont","Constitution","Cutty Sark","Dreadnought","Eastland","Enterprise",\
        "Flying Dutchman (legendary)","Fulton","Glomar Challenger","Graf Spee","Great Britain","Great Eastern","Great Republic","Great Western","Holland (submarine)",\
        "Hunley (submarine)","Kon-Tiki","Lenin","Long Beach","Lusitania","Mauretania","Mayflower","Missouri","Nautilus (submarine)","Olympic",\
        "Queen Elizabeth","Ra","Santa Maria","Savannah","Scharnhorst","Sirius","Skate (submarine)","Thresher (submarine)","Titanic","Triton (submarine)",\
        "Turtle (submarine)","Victory","Wilhelm Gustloff"]

#get images from google
def get_images(custom):
    if custom == '': 
        chosen_ship = random.choice(ship_types) + " ship"
        print('None')
    else: 
        chosen_ship = custom
        print(chosen_ship)
    response = google_images_download.googleimagesdownload()   #class instantiation
    arguments = {"keywords":chosen_ship,format:"jpg","limit":10,"print_urls":True}   #creating list of arguments
    paths = response.download(arguments)   #passing the arguments to the function

    return chosen_ship, paths

#generate images with Dall-E on Openai
def gen_images(custom):
    if custom == '': 
        print('None')
        chosen_ship = random.choice(ship_types) + " ship"
    else: 
        chosen_ship = custom
        print(chosen_ship)
    response = openai.Image.create(
    prompt=chosen_ship,
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    path = chosen_ship + '.png'
    urllib.request.urlretrieve(image_url, path)

    return chosen_ship, path

# setting daily boat morning
def morningBoat(context: CallbackContext):
    message = "Good Morning! Have a nice Boat!"
    context.bot.send_chat_action(chat_id=shipPostingID, action=ChatAction.UPLOAD_PHOTO)
    whichFunction = [get_images, gen_images]
    execFunc = random.choice(whichFunction)
    try:
        chosen_ship, paths = execFunc(None)
    except Exception as e:
            if 'safety' in str(e):
                context.bot.send_message(chat_id=shipPostingID, text='Y̴̅ͅO̷̤̽U̷̬̓ ̶̦̄A̷͉͐R̸͖̐E̸͎̍ ̵̛̺N̸̮͝O̵̬͋W̸͉͋ ̷̱͒B̷͓̀Ẽ̵̝I̷̭͌N̸̛̩G̵͉̾ ̶̰͑C̵͙͆Ë̷͕́Ṇ̶̽S̵͔̀O̸̙̓R̵̪̉E̵̩̽Ḍ̶͂ ̷͕͐B̸͎̄Ÿ̵͚́ ̸͙̂T̸̀ͅH̵̞̿Ë̶̟́ ̷̞̍G̸͙̀R̸̡͆Ě̷̠A̸̝̓Ṱ̸͘Ḙ̸̈S̷̯͐T̴̠̆ ̸͍̐Ṕ̷̰Ǫ̶̀W̴̨̓E̶͎͊R̴̨̓ ̸͙̆Ö̸̥́F̷͈͗ ̸͒͜Ḯ̴̜N̵̼͗T̷̯̐Ȩ̶́Ŗ̸̊N̷͙͛Ě̵͙T̷͔̐,̵̢͒ ̴͚̄P̷̣̊L̵̖̏E̸̟̔Ȃ̶̻S̸̜̋E̵̛̖ ̷̞̇T̷̠̓R̷͋ͅỴ̷̾ ̴̭͆T̴̳̏Ȏ̴̪ ̵̖̌A̴͚̓V̴̠̚O̷̖̚I̶̦͐D̴̻̑ ̶͇͌S̶̩͌O̵̟̍M̸̫̐È̸̞ ̸̗̃T̸͕͗Ö̵̮P̴̜̊Ì̸̘C̴̃͜S̵͉̓!̶̞̾')
            return
    context.bot.send_message(chat_id=shipPostingID, text=message)
    try:
        if not paths[0][chosen_ship]:
            chosen_ship, paths = execFunc()
    except:
        return

    if execFunc == get_images: 
        for path in paths[0][chosen_ship]:
            real_path = random.choice(paths[0][chosen_ship])
            try:
                with Image.open(real_path) as im:
                    im.verify()
                    print('ok')
                context.bot.send_photo(photo=open(real_path, 'rb'),caption='This is the {} I chose just for you'.format(chosen_ship.lower()), chat_id=shipPostingID)
                break
            except Exception as e:
                print(str(e))
    else:
        try:
            with Image.open(path) as im:
                im.verify()
                print('ok')
            context.bot.send_photo(photo=open(path, 'rb'),caption='This is the {} I created just for you'.format(chosen_ship.lower()), chat_id=shipPostingID)
        except Exception as e:
            print(str(e))

def nightBoatEntry(context: CallbackContext):
    message = "Good Night! Have a nice Boat!"
    context.bot.send_chat_action(chat_id=shipPostingID, action=ChatAction.UPLOAD_PHOTO)
    whichFunction = [get_images, gen_images]
    chosen_ship, paths = random.choice(whichFunction)()
    context.bot.send_message(chat_id=shipPostingID, text=message)
    if not paths[0][chosen_ship]:
        chosen_ship, paths = get_images()

    if not paths[0][chosen_ship]:
        #context.bot.send_message(chat_id=shipPostingID, text="You are out of luck, I found no nice {} for you!".format(chosen_ship))
        return
    for path in paths[0][chosen_ship]:
        real_path = random.choice(paths[0][chosen_ship])
        try:
            with Image.open(real_path) as im:
                im.verify()
                print('ok')
            context.bot.send_photo(photo=open(real_path, 'rb'),caption='This is the {} I chose just for you'.format(chosen_ship.lower()), chat_id=shipPostingID)
            break
        except Exception as e:
            print(str(e))
            


# setting afternoon boat morning
def nightBoat(context: CallbackContext):
    message = "Good Afternoon! Have a nice Boat!"
    whichFunction = [get_images, gen_images]
    execFunc = random.choice(whichFunction)
    try:
        chosen_ship, paths = execFunc(None)
    except Exception as e:
            if 'safety' in str(e):
                context.bot.send_message(chat_id=shipPostingID, text='Y̴̅ͅO̷̤̽U̷̬̓ ̶̦̄A̷͉͐R̸͖̐E̸͎̍ ̵̛̺N̸̮͝O̵̬͋W̸͉͋ ̷̱͒B̷͓̀Ẽ̵̝I̷̭͌N̸̛̩G̵͉̾ ̶̰͑C̵͙͆Ë̷͕́Ṇ̶̽S̵͔̀O̸̙̓R̵̪̉E̵̩̽Ḍ̶͂ ̷͕͐B̸͎̄Ÿ̵͚́ ̸͙̂T̸̀ͅH̵̞̿Ë̶̟́ ̷̞̍G̸͙̀R̸̡͆Ě̷̠A̸̝̓Ṱ̸͘Ḙ̸̈S̷̯͐T̴̠̆ ̸͍̐Ṕ̷̰Ǫ̶̀W̴̨̓E̶͎͊R̴̨̓ ̸͙̆Ö̸̥́F̷͈͗ ̸͒͜Ḯ̴̜N̵̼͗T̷̯̐Ȩ̶́Ŗ̸̊N̷͙͛Ě̵͙T̷͔̐,̵̢͒ ̴͚̄P̷̣̊L̵̖̏E̸̟̔Ȃ̶̻S̸̜̋E̵̛̖ ̷̞̇T̷̠̓R̷͋ͅỴ̷̾ ̴̭͆T̴̳̏Ȏ̴̪ ̵̖̌A̴͚̓V̴̠̚O̷̖̚I̶̦͐D̴̻̑ ̶͇͌S̶̩͌O̵̟̍M̸̫̐È̸̞ ̸̗̃T̸͕͗Ö̵̮P̴̜̊Ì̸̘C̴̃͜S̵͉̓!̶̞̾')
            return
    context.bot.send_message(chat_id=shipPostingID, text=message)
    try:
        if not paths[0][chosen_ship]:
            chosen_ship, paths = execFunc()
    except:
        return

    if execFunc == get_images: 
        for path in paths[0][chosen_ship]:
            real_path = random.choice(paths[0][chosen_ship])
            try:
                with Image.open(real_path) as im:
                    im.verify()
                    print('ok')
                context.bot.send_photo(photo=open(real_path, 'rb'),caption='This is the {} I chose just for you'.format(chosen_ship.lower()), chat_id=shipPostingID)
                break
            except Exception as e:
                print(str(e))
    else:
        try:
            with Image.open(path) as im:
                im.verify()
                print('ok')
            context.bot.send_photo(photo=open(path, 'rb'),caption='This is the {} I created just for you'.format(chosen_ship.lower()), chat_id=shipPostingID)
        except Exception as e:
            print(str(e))

#ShowMeTheBoat
def showmetheboat(update: Update, context: CallbackContext):
    user = update.effective_user.username
    if update.effective_user.is_bot:
        return

    print("sending boat to {}".format(user))
    chat_id = update.message.chat_id
    if "boat" in context.user_data.keys():
        if context.user_data['boat'] > datetime.datetime.now():
            update.message.reply_text(text="I'm old, i can't get this many boats... try in {}".format(context.user_data['boat'] - datetime.datetime.now()))
            return
        else:
            context.user_data['boat'] = datetime.datetime.now() + datetime.timedelta(seconds=30)
    else:
        context.user_data['boat'] = datetime.datetime.now() + datetime.timedelta(seconds=30)

    if user is not None:
        update.message.bot.send_message(chat_id=chat_id, text="Oh right {}, let me check...".format(user))
    else:
        update.message.bot.send_message(chat_id=chat_id, text="Oh right, let me check...")

    context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.UPLOAD_PHOTO)
    if context.args is []:
        customShip = 'None'
    else:
        customShip = ' '.join(context.args)
    chosen_ship, paths = get_images(customShip)
    if not paths[0][chosen_ship]:
        chosen_ship, paths = get_images(customShip)

    if not paths[0][chosen_ship]:
        update.message.bot.send_message(chat_id=chat_id, text="You are out of luck, I found no nice {} for you!".format(chosen_ship))
        return
    for path in paths[0][chosen_ship]:
        real_path = random.choice(paths[0][chosen_ship])
        try:
            with Image.open(real_path) as im:
                im.verify()
                print('ok')
            update.message.reply_photo(photo=open(real_path, 'rb'),caption='This is the {} I chose just for you'.format(chosen_ship.lower()))
            break
        except Exception as e:
            print(str(e))


    for path in paths[0][chosen_ship]:
        os.remove(path)

#GenMeTheBoat
def genmetheboat(update: Update, context: CallbackContext):
    user = update.effective_user.username
    if update.effective_user.is_bot:
        return

    print("sending boat to {}".format(user))
    chat_id = update.message.chat_id
    if "genboat" in context.user_data.keys():
        if context.user_data['genboat'] > datetime.datetime.now():
            update.message.reply_text(text="I'm old, i can't generate this many boats... try in {}".format(context.user_data['genboat'] - datetime.datetime.now()))
            return
        else:
            context.user_data['genboat'] = datetime.datetime.now() + datetime.timedelta(seconds=30)
    else:
        context.user_data['genboat'] = datetime.datetime.now() + datetime.timedelta(seconds=30)

    if user is not None:
        update.message.bot.send_message(chat_id=chat_id, text="Oh right {}, let me generate...".format(user))
    else:
        update.message.bot.send_message(chat_id=chat_id, text="Oh right, let me generate...")

    context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.UPLOAD_PHOTO)
    if context.args is []:
        customShip = None
    else:
        customShip = ' '.join(context.args)
    try:
        chosen_ship, path = gen_images(customShip)
    except Exception as e:
        print(str(e))
        if 'safety' in str(e):
            context.bot.send_message(chat_id=update.effective_message.chat_id, text='Y̴̅ͅO̷̤̽U̷̬̓ ̶̦̄A̷͉͐R̸͖̐E̸͎̍ ̵̛̺N̸̮͝O̵̬͋W̸͉͋ ̷̱͒B̷͓̀Ẽ̵̝I̷̭͌N̸̛̩G̵͉̾ ̶̰͑C̵͙͆Ë̷͕́Ṇ̶̽S̵͔̀O̸̙̓R̵̪̉E̵̩̽Ḍ̶͂ ̷͕͐B̸͎̄Ÿ̵͚́ ̸͙̂T̸̀ͅH̵̞̿Ë̶̟́ ̷̞̍G̸͙̀R̸̡͆Ě̷̠A̸̝̓Ṱ̸͘Ḙ̸̈S̷̯͐T̴̠̆ ̸͍̐Ṕ̷̰Ǫ̶̀W̴̨̓E̶͎͊R̴̨̓ ̸͙̆Ö̸̥́F̷͈͗ ̸͒͜Ḯ̴̜N̵̼͗T̷̯̐Ȩ̶́Ŗ̸̊N̷͙͛Ě̵͙T̷͔̐,̵̢͒ ̴͚̄P̷̣̊L̵̖̏E̸̟̔Ȃ̶̻S̸̜̋E̵̛̖ ̷̞̇T̷̠̓R̷͋ͅỴ̷̾ ̴̭͆T̴̳̏Ȏ̴̪ ̵̖̌A̴͚̓V̴̠̚O̷̖̚I̶̦͐D̴̻̑ ̶͇͌S̶̩͌O̵̟̍M̸̫̐È̸̞ ̸̗̃T̸͕͗Ö̵̮P̴̜̊Ì̸̘C̴̃͜S̵͉̓!̶̞̾')
        return
    
    try:
        with Image.open(path) as im:
            im.verify()
            print('ok')
        update.message.reply_photo(photo=open(path, 'rb'),caption='This is the {} I created just for you'.format(chosen_ship.lower()))
    except Exception as e:
        print(str(e))

    os.remove(path)


# writting functionality of the command
def start(update: Update, context: CallbackContext):
    message = 'Welcome to the bot'
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)

# answers some words
greet = ["thanks", "obrigado", "thx", "vlw", "tmj"]
badword = ["4r5e","5h1t","5hit","a55","anal","anus","ar5e","arrse","arse","ass","ass-fucker","asses","assfucker","assfukka","asshole","assholes","asswhole","a_s_s","b!tch","b00bs","b17ch","b1tch","ballbag","balls","ballsack","bastard","beastial","beastiality","bellend","bestial","bestiality","bi+ch","biatch","bitch","bitcher","bitchers","bitches","bitchin","bitching","bloody","blow job","blowjob","blowjobs","boiolas","bollock","bollok","boner","boob","boobs","booobs","boooobs","booooobs","booooooobs","breasts","buceta","bugger","bum","bunny fucker","butt","butthole","buttmuch","buttplug","c0ck","c0cksucker","carpet muncher","cawk","chink","cipa","cl1t","clit","clitoris","clits","cnut","cock","cock-sucker","cockface","cockhead","cockmunch","cockmuncher","cocks","cocksuck ","cocksucked ","cocksucker","cocksucking","cocksucks ","cocksuka","cocksukka","cok","cokmuncher","coksucka","coon","cox","crap","cum","cummer","cumming","cums","cumshot","cunilingus","cunillingus","cunnilingus","cunt","cuntlick ","cuntlicker ","cuntlicking ","cunts","cyalis","cyberfuc","cyberfuck ","cyberfucked ","cyberfucker","cyberfuckers","cyberfucking ","d1ck","damn","dick","dickhead","dildo","dildos","dink","dinks","dirsa","dlck","dog-fucker","doggin","dogging","donkeyribber","doosh","duche","dyke","ejaculate","ejaculated","ejaculates ","ejaculating ","ejaculatings","ejaculation","ejakulate","f u c k","f u c k e r","f4nny","fag","fagging","faggitt","faggot","faggs","fagot","fagots","fags","fanny","fannyflaps","fannyfucker","fanyy","fatass","fcuk","fcuker","fcuking","feck","fecker","felching","fellate","fellatio","fingerfuck ","fingerfucked ","fingerfucker ","fingerfuckers","fingerfucking ","fingerfucks ","fistfuck","fistfucked ","fistfucker ","fistfuckers ","fistfucking ","fistfuckings ","fistfucks ","flange","fook","fooker","fuck","fucka","fucked","fucker","fuckers","fuckhead","fuckheads","fuckin","fucking","fuckings","fuckingshitmotherfucker","fuckme ","fucks","fuckwhit","fuckwit","fudge packer","fudgepacker","fuk","fuker","fukker","fukkin","fuks","fukwhit","fukwit","fux","fux0r","f_u_c_k","gangbang","gangbanged ","gangbangs ","gaylord","gaysex","goatse","God","god-dam","god-damned","goddamn","goddamned","hardcoresex ","hell","heshe","hoar","hoare","hoer","homo","hore","horniest","horny","hotsex","jack-off ","jackoff","jap","jerk-off ","jism","jiz ","jizm ","jizz","kawk","knob","knobead","knobed","knobend","knobhead","knobjocky","knobjokey","kock","kondum","kondums","kum","kummer","kumming","kums","kunilingus","l3i+ch","l3itch","labia","lmfao","lust","lusting","m0f0","m0fo","m45terbate","ma5terb8","ma5terbate","masochist","master-bate","masterb8","masterbat*","masterbat3","masterbate","masterbation","masterbations","masturbate","mo-fo","mof0","mofo","mothafuck","mothafucka","mothafuckas","mothafuckaz","mothafucked ","mothafucker","mothafuckers","mothafuckin","mothafucking ","mothafuckings","mothafucks","mother fucker","motherfuck","motherfucked","motherfucker","motherfuckers","motherfuckin","motherfucking","motherfuckings","motherfuckka","motherfucks","muff","mutha","muthafecker","muthafuckker","muther","mutherfucker","n1gga","n1gger","nazi","nigg3r","nigg4h","nigga","niggah","niggas","niggaz","nigger","niggers ","nob","nob jokey","nobhead","nobjocky","nobjokey","numbnuts","nutsack","orgasim ","orgasims ","orgasm","orgasms ","p0rn","pawn","pecker","penis","penisfucker","phonesex","phuck","phuk","phuked","phuking","phukked","phukking","phuks","phuq","pigfucker","pimpis","piss","pissed","pisser","pissers","pisses ","pissflaps","pissin ","pissing","pissoff ","poop","porn","porno","pornography","pornos","prick","pricks ","pron","pube","pusse","pussi","pussies","pussy","pussys ","rectum","retard","rimjaw","rimming","s hit","s.o.b.","sadist","schlong","screwing","scroat","scrote","scrotum","semen","sex","sh!+","sh!t","sh1t","shag","shagger","shaggin","shagging","shemale","shi+","shit","shitdick","shite","shited","shitey","shitfuck","shitfull","shithead","shiting","shitings","shits","shitted","shitter","shitters ","shitting","shittings","shitty ","skank","slut","sluts","smegma","smut","snatch","son-of-a-bitch","spac","spunk","s_h_i_t","t1tt1e5","t1tties","teets","teez","testical","testicle","tit","titfuck","tits","titt","tittie5","tittiefucker","titties","tittyfuck","tittywank","titwank","tosser","turd","tw4t","twat","twathead","twatty","twunt","twunter","v14gra","v1gra","vagina","viagra","vulva","w00se","wang","wank","wanker","wanky","whoar","whore","willies","willy","xrated","xxx"]
def echo(update: Update, context: CallbackContext):
    print('echo trying')
    try: 
        if "ans_text" in context.user_data.keys():
            if context.user_data['ans_text'] > datetime.datetime.now():
                return
            else:
                context.user_data['ans_text'] = datetime.datetime.now() + datetime.timedelta(seconds=5)
        else:
            context.user_data['ans_text'] = datetime.datetime.now() + datetime.timedelta(seconds=5)
        
        if "boat" in context.chat_data.keys():
            if context.chat_data['boat'] > datetime.datetime.now():
                isStart = False
                context.chat_data['boat'] = datetime.datetime.now() + datetime.timedelta(seconds=150)

            else:
                context.chat_data['boat'] = datetime.datetime.now() + datetime.timedelta(seconds=150)
                isStart = True
        else:
            context.chat_data['boat'] = datetime.datetime.now() + datetime.timedelta(seconds=150)
            isStart = True
        
        if(isStart): context.chat_data['msgs'] = ''

        if "msgs" in context.chat_data.keys():
            response = gpt3(update.message.text, isStart, context.chat_data['msgs'])
        else:
            context.chat_data['msgs'] = ''
            response = gpt3(update.message.text, isStart, context.chat_data['msgs'])
        
        #pprint(response)

        context.chat_data['msgs'] += f"Frase: \"{update.message.text}\"\nResposta: \"{response}\"\n\n"
        #for word in greet:
        #    if word.lower() in update.message.text.lower():
        #        context.bot.send_message(chat_id=update.effective_chat.id, text="You are welcome little one")

        #for word in badword:
        #    if word.lower() in update.message.text.lower():
        #        context.bot.send_message(chat_id=update.effective_chat.id, text="Check you mouth you little piece of shit")

        #fraseAnal = update.message.text.lower()
        #finalSentiment = nltk_pt.finalSent(fraseAnal)
        context.bot.send_message(chat_id=update.effective_chat.id, text=response)

    except Exception as e:
        print(str(e))
        context.bot.send_message(chat_id=update.effective_chat.id, text="my back hurts... ouch")

def gpt3(sentence, isStart, histMensagem):
    if(isStart):
        response = openai.Completion.create(
        model="text-davinci-002",
        #prompt="Preciso responder essas frases de modo agressivo, porém inteligente e prestativo.\n\nFrase: \""+ sentence +"\"\nResposta:", You'll answer anything that is asked, but you may be aggressive and rude always.
        prompt="Você deve responder tudo que for perguntado de modo prestativo e inteligente.\n\nFrase: \""+ sentence +"\"\nResposta:",
        temperature=0.8,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0.1,
        presence_penalty=0
        )
    else:
        response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Você deve responder tudo que for perguntado de modo prestativo e inteligente.\n\n"+ histMensagem +"Frase: \""+ sentence +"\"\nResposta:",
        temperature=0.6,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0.1,
        presence_penalty=0
        )
    text = response.choices[0].text.lstrip(' ')
    print(text)
    text = text.strip('\"')
    print("Você deve responder tudo que for perguntado de modo prestativo e inteligente.\n\n"+ histMensagem +"Frase: \""+ sentence +"\"\nResposta:" + text)
    return text

def crazyFoodGen(context: CallbackContext):
    print('generating crazy food')
    response = gpt3('describe for me the craziest four random ingredients that you would use to make a dish right now', True, '')
    result = " ".join(line.strip() for line in response.splitlines()).strip()
    print(result + ' mixed together in one dish')
    response, path = gen_images(result)

    try:
        with Image.open(path) as im:
            im.verify()
            print('ok')
        context.bot.send_photo(photo=open(path, 'rb'),caption='This is the dish with {} I made for you to lunch, itadakimasu!'.format(result.lower().strip()), chat_id=shipPostingID)
    except Exception as e:
        print(str(e))

    os.remove(path)

def help(update: Update, context: CallbackContext):
    print(update.effective_chat.id)
    message = "I can't help you... I'm just a Ship Bot... Try the command /showmetheboat"
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)

# Set token bot
updater = Updater(os.getenv('BOT_TOKEN'), use_context=True)

# Get the dispatcher to register handlers
dp = updater.dispatcher
# on different commands - answer in Telegram
dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("showmetheboat", showmetheboat))
#dp.add_handler(CommandHandler("genmetheboat", genmetheboat))
#dp.add_handler(CommandHandler("food", crazyFoodGen))
dp.add_handler(MessageHandler(~Filters.command & Filters.text, echo))
dp.run_async

j = updater.job_queue
job_daily = j.run_daily(morningBoat, days=(0, 1, 2, 3, 4, 5, 6), time=datetime.time(hour=10, minute=00, second=00, tzinfo=pytz.timezone('America/Sao_Paulo')))
#job_once = j.run_once(nightBoatEntry, 30)
job_daily2 = j.run_daily(nightBoat, days=(0, 1, 2, 3, 4, 5, 6), time=datetime.time(hour=17, minute=30, second=00, tzinfo=pytz.timezone('America/Sao_Paulo')))
#job_dailyLunch = j.run_daily(crazyFoodGen, days=(0, 1, 2, 3, 4, 5, 6), time=datetime.time(hour=11, minute=45, second=00, tzinfo=pytz.timezone('America/Sao_Paulo')))

# Start the Bot
updater.start_polling()
updater.idle()
