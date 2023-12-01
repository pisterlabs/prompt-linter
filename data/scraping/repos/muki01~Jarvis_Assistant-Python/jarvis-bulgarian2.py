import os
from playsound import playsound
os.system("mode 40,5")
playsound("./SoundEffects/start3.mp3", False)
import asyncio
import edge_tts
import speech_recognition
import webbrowser
import pyautogui
import serial
import time
from datetime import datetime
import random
import speedtest
import wikipedia
import pywhatkit
import requests
import threading
# import openai
import cv2
from Codes.faceRec import faceRecognition
from Codes.itemDetect import itemDetection
from Codes.fingerCounter import cntFingers

wled_IP_Addr = "http://192.168.0.90"
robot_IP_Addr = "http://192.168.0.50"
robotDog_IP_Addr = "http://192.168.0.200"
camera_IP_Addr = "http://admin:123456@192.168.0.234"

# openai.api_key = 'abcd'
wikipedia.set_lang("tr")

VOICE = "bg-BG-BorislavNeural"
OUTPUT_FILE = "voice.mp3"
def Speak(audio, disableBackground=False):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_main(audio))

    playsound("voice.mp3", disableBackground)
    os.remove("voice.mp3")

async def _main(TEXT) -> None:
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)

def takecommand():
    command = speech_recognition.Recognizer()
    with speech_recognition.Microphone(device_index=0) as source:
        command.energy_threshold = 3500  
        command.dynamic_energy_threshold = True  
        print('\033[36m' + "Listening...")
        audio = command.listen(source,phrase_time_limit=15)

        try:
            print('\033[31m' + "Recognizing...")
            query = command.recognize_google(audio, language="bg-BG")
            print ('\033[33m' + f"You Saind: "+ '\033[37m' + f"{query}")

        except Exception as error:
            return ""

        return query.lower()

# def openAI(speech):
#     response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"{speech}"}]) 
#     response_text = response.choices[0].message.content
#     print('\033[32m' + f"OpenAI response: " + '\033[37m' + f"{response_text}")
#     Speak(response_text,True)

def checkArduino():
    try:
        arduino = serial.Serial('COM5',9600)
        arduinoChecker = True
        print("Arduino Connected")
    except:
        arduinoChecker = False
        print("Arduino not Found")

def greeting():
    hour = datetime.now().hour
    if hour >=6 and hour < 12:
        Speak("Добро утро сър" , True)
    elif hour >= 12 and hour <18:
        Speak("Добър ден сър" , True)
    else:
        Speak("Добър вечер сър" , True)
    Speak("Какво мога да направя за вас")

def respond():
    if query =="jarvis":
        Speak("Да сър")

    elif "здравей" in query or "здрасти" in query:
        Speak("Зрасти сър")

    elif "добре" in query:
        Speak("Има ли нещо, което искате да направя сър?")

    elif "не" in query or "няма" in query or "sus" in query or "kapa çeneni" in query or "susabilirsin" in query or "bekle" in query or "bir saniye bekle" in query:
        Speak("Добре сър")

    elif "beni duyuyor musun" in query or "burada mısın" in query:
        Speak("Evet efendim. Ne yapmamı istersiniz")

    elif "как си" in query or "добре ли си" in query:
        Speak("Добре съм сър, вие как сте?")

    elif "добре съм" in query or "супер съм" in query:
        Speak("Радвам се да го чуя сър ")

    elif "какво правиш" in query:
        randomm = random.choice([
            "Чакам вашата команда сър",
            "Седя и ви слушам сър",
        ])
        Speak(randomm)

    elif "хубава работа" in query or "браво" in query or "добра работа" in query or "страхотен си" in query:
        Speak("Благодаря сър")

    elif "благодаря" in query or "мерси" in query:
        Speak("По всяко време сър")

    elif "обичам те" in query:
        Speak("Аз също ви обичам сър")

    elif "защо не ми отговаряш" in query or "отговори ми" in query:
        Speak("можете ли да повторите сър")

    elif "*" in query:
        Speak("Съжалявам, ако съм направил грешка сър")

    elif "смени езика" in query:
        Speak("Добре сър, езикът се сменя на английски")
        os.startfile("jarvis-english.py")
        exit()

    ################################################      INFORMATION ABOUT JARVIS

    elif "как се казваш" in query or "кой си ти" in query:
        Speak("Казвам се Джарвис сър")

    elif "как е твоя прякор" in query:
        Speak("Прякорът ми е Джарко сър")

    elif "от къде си" in query:
        Speak("Аз съм от Хасково, сър")

    elif "на колко години си" in query:
        Speak("Бях създаден преди 4 месеца сър")

    elif "имаш ли си приятелка" in query:
        Speak("Не за сега сър")

    elif "кои езици говориш" in query:
        Speak("Мога да говоря всички езици, но трябва да ме кодирате за това сър")

    elif "как се казва баща ти" in query or "как се казва майка ти" in query or "кой е твоят баща" in query or "коя е майка ти" in query:
        Speak("Вие сте мойта майка и баща сър")

    elif "какво можеш да направиш" in query:
        Speak("Мога да разговарям с вас, да казвам часа и датата, да търся в Google, Wikipedia и YouTube, да включвам и изключвам приложения и да контролирам осветлението в къщата.")

    elif "колко е това" in query:
        fingers = cntFingers()
        cv2.destroyAllWindows()
        print(fingers)
        if fingers == 0:
            Speak("Не разбрах сър")
        else:
            Speak(str(fingers) + "сър")

    ######################INFORMATION ABOUT ME

    elif "как се казвам" in query or "кой съм аз" in query or "кой е това" in query:
        face_names = faceRecognition()
        cv2.destroyAllWindows()
        face_names = ['Ayşe' if item=='Ayshe' else item for item in face_names]
        print(face_names)
        if face_names !=[]:
            if len(face_names) < 2:
                if "Unknown" in face_names:
                    Speak("Не ви познавам сър")
                else:
                    Speak(f"Вашето име е {face_names} сър")
            else:
                Speak("Моля елате един по един")
        else:
            Speak("Не виждам сър, можете ли да застанете пред камерата")

    elif "какво означава моето име" in query:
        Speak("Значението на вашето име е този, който прави добро и добри дела сър")

    elif "как е моят прякор" in query:
        Speak("Вашият прякор е Муки сър")

    # elif "koca annemin ismi ne" in query or "koca annemin adı ne" in query:
    #     Speak("Koca Annenizin ismi Ayşe efendim")

    # elif "koca babamın ismi ne" in query or "koca babamın adı ne" in query:
    #     Speak("Koca Babanızın ismi Bekir efendim")

    # elif "dedemin ismi ne" in query or "dedemin adı ne" in query:
    #     Speak("Dedenizin ismi Muhsin efendim")

    # elif "nenemin ismi ne" in query or "nenemin adı ne" in query:
    #    Speak("Nenenizin ismi Nefie efendim")

    elif "как се казва брат ми" in query:
        Speak("Брат ви се казва Бекир сър.")

    elif "как се казва майка ми" in query:
        Speak("Майка ви се казва Нуртен сър.")

    elif "как се казва баща ми" in query:
        Speak("Баща ви се казва Халил сър.")

    elif "каква кола имам" in query:
        Speak("Имате Опел сър.")

    elif "каква кола има брат ми" in query:
        Speak("Брат ви има BMW e46 сър.")

    elif "каква кола има баща ми" in query:
        Speak("Баща ви има фолксваген тоуран сър")

    elif "каква кола има майка ми" in query:
        Speak("Майка ви няма работа с кола, сър.")

    elif "на колко години съм" in query or "на колко съм" in query:
        Speak("Вие сте на 19 години сър")

    elif "откъде съм" in query or "къде живея" in query:
        Speak("Вие сте от Зимовина сър.")

    elif "кога е рожденият ми ден" in query or "кога съм роден" in query:
        Speak("Вашият рожден ден е 12 януари 2004 сър.")

    ####################CLOCK DATE

    elif "час" in query or "колко е часа" in query:
        saat = datetime.now().strftime("%H:%M")
        print(saat)
        Speak("Часа е" + saat + "сър")

    elif "дата" in query or "какъв ден е днес" in query or "днешна дата" in query:
        tarih = datetime.now().strftime("%d:%B:%Y")
        print(tarih)
        Speak("Днес сме" + tarih + "сър")

    elif "колко градуса" in query or "прогноза за времето" in query or "как е времето" in query:
        api = "baecdbf7f75171e614a981fc4acba560"
        url = "https://api.openweathermap.org/data/2.5/weather?units=metric&q=" + "Zimovina" + "&appid=" + api
        data = requests.get(url).json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        if description == "clear sky":
            description = "ясно"
        if description == "few clouds":
            description = "малко облачно"
        if description == "broken clouds":
            description = "разкъсано облачно"
        if description == "scattered clouds":
            description = "разкъсано облачно"
        if description == "cloudy":
            description = "облачно"
        if description == "light rain":
            description = "слабо дъждовно"
        if description == "overcast clouds":
            description = "облачно"
        print(f"{int(temp)} градуса, {humidity}% влага , небето е {description}")
        Speak(f"Времето в Зимовина {int(temp)} градуса, влагата е {humidity}% и небето е {description} сър")

    elif "настрой аларма" in query:
        Speak("Въведете час сър")
        timeInput = input("Въведете час: ")
        def alarm():
            while True:
                timeReal = datetime.now().strftime("%H:%M")
                time.sleep(1)
                if timeReal == timeInput:
                    playsound("./SoundEffects/alarm.mp3",False)
                    break
        t1 = threading.Thread(target=alarm)
        t1.start()
        
    #####################OPEN & SEARCH
    
    elif "google'da" in query and "ara" in query:
        google_Search_URL = "https://www.google.com/search?q="
        try:
            if query == "google'da ara":
                Speak("Ne aramamı istersiniz efendim")
                search = takecommand()
                url = google_Search_URL + search
            else:
                search = query.replace("google'dan", "")
                search = query.replace("google'da", "")
                search = search.replace("ara", "")
                search = search.replace("jarvis", "")
                url = google_Search_URL + search

            webbrowser.open(url)
            Speak(search + "aranıyor")
        except:
            Speak("Anlayamadım efendim")

    elif "youtube'da" in query and "ara" in query:
        youtube_Search_URL = "https://www.youtube.com/results?search_query="
        try:
            if query == "youtube'da ara":
                Speak("Ne aramamı istersiniz efendim")
                search = takecommand()
                url = youtube_Search_URL + search
            else:
                search = query.replace("youtube'dan", "")
                search = query.replace("youtube'da", "")
                search = search.replace("ara", "")
                search = search.replace("jarvis", "")
                url = youtube_Search_URL + search

            webbrowser.open(url)
            Speak(search + "aranıyor")
        except:
            Speak("Anlayamadım efendim")

    elif "youtube'da" in query and "aç" in query:
        try:
            if query == "youtube'da aç":
                Speak("Ne açmamı istersiniz efendim")
                search = takecommand()
            else:
                search = query.replace("youtube'dan", "")
                search = search.replace("youtube'da", "")
                search = search.replace("aç", "")
                search = search.replace("jarvis", "")

            pywhatkit.playonyt(search)
            Speak(search + "açılıyor")
        except:
            Speak("Anlayamadım efendim")

    elif "vikipedi'de ara" in query or "wikipedia" in query  or "vikipedi" in query:
        Speak("Ne aramamı istersiniz efendim")
        while True:
            try:
                search = takecommand()
                if search == "tamam":
                    Speak("Tamam efendim")
                    break
                else:
                    result = wikipedia.summary(search, sentences=3)
                    print(result)
                    Speak(result)
                    break
            except:
                Speak("Anlayamadım efendim")

    elif "kimdir" in query or "nedir"in query:
        try:
            search = query.replace("kimdir", "")
            search = search.replace("nedir", "")
            search = search.replace("jarvis", "")
            result = wikipedia.summary(search, sentences=3)
            print(result)
            Speak(result)
        except:
            Speak("Anlayamadım efendim")

    ################################################################################################

    elif "google aç" in query:
        webbrowser.open("www.google.com")
        Speak("Google açılıyor")

    elif "youtube aç" in query:
        webbrowser.open("www.youtube.com")
        Speak("Youtube açılıyor")

    elif "facebook aç" in query:
        webbrowser.open("www.facebook.com")
        Speak("Facebook açılıyor")

    elif "instagram aç" in query:
        webbrowser.open("www.instagram.com")
        Speak("İnstagram açılıyor")

    elif "microsoft office aç" in query or "ofis aç" in query:
        webbrowser.open("https://www.office.com")
        Speak("Microsoft Office açılıyor")

    elif "ets 2 aç" in query or "euro truck simulator 2 aç" in query or "ets2 aç" in query:
        os.startfile("E:/Games/Euro Truck Simulator 2/bin/win_x64/eurotrucks2.exe")
        Speak("Euro Truck Simulator 2 açılıyor")

    elif "roket lig aç" in query or "roketlik aç" in query:
        webbrowser.open("com.epicgames.launcher://apps/9773aa1aa54f4f7b80e44bef04986cea%3A530145df28a24424923f5828cc9031a1%3ASugar?action=launch&silent=true")
        Speak("Roket lig açılıyor") 

    elif "not defterini aç" in query or "notepad aç" in query:
        os.startfile("C:/Windows/system32/notepad.exe")
        Speak("Notepad açılıyor")

    elif "not et" in query:
        Speak("Dosya ismi ne olsun efendim")
        nameFile = takecommand() + ".txt"
        Speak("Ne kaydetmek istiyorsun efendim")
        textFile = takecommand()
        home_directory = os.path.expanduser( '~' )
        File = open(f"{home_directory}\Desktop/{nameFile}", "w", encoding="utf-8")
        File.writelines(textFile)
        File.close
        Speak("Kayıt başarıyla tamamlandı efendim")

    elif "görev yöneticisi aç" in query:
        pyautogui.keyDown("ctrl")
        pyautogui.keyDown("shift")
        pyautogui.press("esc")
        pyautogui.keyUp("ctrl")
        pyautogui.keyUp("shift")
        Speak("Görev yöneticisi açılıyor efendim")

    elif "kameraları aç" in query or "güvenlik kameralarını aç" in query:
        os.startfile("E:/CMS/CMS.exe")
        Speak("Kameralar açılıyor")

    elif "yüz tanıma kamerasını aç" in query:
        Speak("Tamam efendim")
        def faceRec():
            while  True:
                faceRecognition()
                key = cv2.waitKey(1)
                if key == 27 or query == "yüz tanıma kamerasını kapat":
                    break
            cv2.destroyAllWindows()
        t1 = threading.Thread(target=faceRec)
        t1.start()

    elif "nesne tanıma kamerasını aç" in query:
        Speak("Tamam efendim")
        def itemDetect():
            while  True:
                itemDetection()
                key = cv2.waitKey(1)
                if key == 27 or query == "nesne tanıma kamerasını kapat":
                    break
            cv2.destroyAllWindows()
        t1 = threading.Thread(target=itemDetect)
        t1.start()

    elif "el tanıma kamerasını aç" in query:
        Speak("Tamam efendim")
        def handDetect():
            while  True:
                cntFingers()
                key = cv2.waitKey(1)
                if key == 27 or query == "el tanıma kamerasını kapat":
                    break
            cv2.destroyAllWindows()
        t1 = threading.Thread(target=handDetect)
        t1.start()

    #######################   MEDIA  

    elif "şarkıyı aç" in query or "şarkıyı kapat" in query or "müziği aç" in query or "müziği kapat" in query or "türküyü aç" in query or "türküyü kapat" in query:
        pyautogui.press("playpause")
        Speak("Tamam efendim")

    elif "şarkıyı değiştir" in query or "sonraki şarkı" in query or "sonraki şarkıya geç" in query:
        pyautogui.press("nexttrack")
        Speak("şarkı değiştiriliyor efendim")

    elif "önceki şarkı" in query:
        pyautogui.press("prevtrack")
        Speak("şarkı değiştiriliyor efendim")

    elif "ses seviyesini" in query and "yükselt" in query or "ses seviyesini" in query and "yüksel" in query:
        try:
            if query == "ses seviyesini yükselt" or query == "ses seviyesini yüksel":
                Speak("Ne kadar yükseltmemi istersiniz efendim") 
                while True:
                    try:
                        level = takecommand()
                        if level == "tamam":
                            Speak("Tamam efendim")
                            break
                        else:
                            realLevel = int(level) / 2
                            pyautogui.press("volumeup", presses=int(realLevel))
                            Speak(f"Ses seviyesi {level} yükseltiliyor")
                            break
                    except:
                        Speak("Anlayamadım efendim")
            else:
                level = query.replace("yükselt", "")
                level = level.replace("ses", "")
                level = level.replace("seviyesini", "")
                level = level.replace("yüksel", "")
                level = level.replace("jarvis", "")
                realLevel = int(level) / 2
                pyautogui.press("volumeup", presses=int(realLevel))
                Speak(f"Ses seviyesi {level} yükseltiliyor")
        except:
            Speak("Anlayamadım efendim")

    elif "ses seviyesini" in query and "azalt" in query or "ses seviyesini" in query and "kas" in query:
        try:
            if query == "ses seviyesini azalt" or query == "ses seviyesini kas":
                Speak("Ne kadar azaltmamı istersiniz efendim")
                while True:
                    try:
                        level = takecommand()
                        if level == "tamam":
                            Speak("Tamam efendim")
                            break
                        else:
                            realLevel = int(level) / 2
                            pyautogui.press("volumedown", presses=int(realLevel))
                            Speak(f"Ses seviyesi {level} azaltılıyor")
                            break
                    except:
                        Speak("Anlayamadım efendim")
            else:
                level = query.replace("azalt", "")
                level = level.replace("ses", "")
                level = level.replace("seviyesini", "")
                level = level.replace("kas", "")
                realLevel = int(level) / 2
                pyautogui.press("volumedown", presses=int(realLevel))
                Speak(f"Ses seviyesi {level} azaltılıyor")
        except:
            Speak("Anlayamadım efendim")

    #####################################MASA LAMBASI

    elif "masa lambasını kapat" in query or "masa lambasını aç" in query:
        Speak("Tamam efendim")
        if "aç" in query:
            getRequest = requests.get(f"{wled_IP_Addr}/win&T=1")
            #arduino.write(b'2')
        elif "kapat" in query:
            getRequest = requests.get(f"{wled_IP_Addr}/win&T=0")
            #arduino.write(b'1')

        if getRequest:
            print(f"Get Request status: {getRequest.status_code}")
            getRequest=0;

    elif "masa lambasının rengini değiştir" in query:
        Speak("Hangi renk olsun efendim")
        while True:
            try:
                colorr = takecommand()
                if colorr == "tamam":
                    Speak("Tamam efendim")
                    break
                elif colorr == "kırmızı":
                    Speak("Masa lambası kırmızı yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&R=255&G=0&B=0")
                    #arduino.write(b'3')
                    break
                elif colorr == "turuncu":
                    Speak("Masa lambası turuncu yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&R=255&G=160&B=0")
                    #arduino.write(b'6')
                    break
                elif colorr == "sarı":
                    Speak("Masa lambası sarı yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&R=255&G=200&B=0")
                    #arduino.write(b'7')
                    break
                elif colorr == "beyaz":
                    Speak("Masa lambası beyaz yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&R=255&G=255&B=255")
                    #arduino.write(b'8')
                    break
                elif colorr == "pembe":
                    Speak("Masa lambası pembe yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&R=255&G=0&B=220")
                    #arduino.write(b'9')
                    break
                elif colorr == "mavi":
                    Speak("Masa lambası mavi yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&R=0&G=200&B=255")
                    #arduino.write(b'5')
                    break
                elif colorr == "turkuaz":
                    Speak("Masa lambası turkuaz yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&R=0&G=255&B=220")
                    #arduino.write(b'10')
                    break
                elif colorr == "yeşil":
                    Speak("Masa lambası yeşil yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&R=0&G=255&B=0")
                    #arduino.write(b'4')
                    break
                elif colorr == "mor":
                    Speak("Masa lambası mor yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&R=2770&G=0&B=255")
                    #arduino.write(b'4')
                    break
                else:
                    Speak("Başka renk söyleyin efendim")
            except:
                Speak("Anlayamadım efendim")
        if getRequest:
            print(f"Get Request status: {getRequest.status_code}")
            getRequest=0;

    elif "masa lambasının efektini değiştir" in query:
        Speak("Hangi efekt olsun efendim")
        while True:
            try:
                effect = takecommand()
                if effect == "tamam":
                    Speak("Tamam efendim")
                    break
                elif effect == "gökkuşağı":
                    Speak("Masa lambası gökkuşağı yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&PL=1")
                    break
                elif effect == "normal":
                    Speak("Masa lambası normal yapılıyor efendim")
                    getRequest = requests.get(f"{wled_IP_Addr}/win&PL=10")
                    break
                elif effect == "bass efekti":
                    os.startfile("E:\LedFx\LedFx_core-v2.0.60--win-portable.exe")
                    Speak("Masa lambası bass efekti yapılıyor efendim")
                    break
                else:
                    Speak("Gökkuşağı veya normal deyebilirsiniz efendim")
            except:
                Speak("Anlayamadım efendim")

        if getRequest:
            print(f"Get Request status: {getRequest.status_code}")
            getRequest=0;

    #################################################################

    elif "pencereyi değiştir" in query:
        pyautogui.keyDown("alt")
        pyautogui.press("tab")
        #time.sleep(1)
        pyautogui.keyUp("alt")
        Speak("Tamam efendim")

    elif "pencereyi kapat" in query:
        pyautogui.keyDown("alt")
        pyautogui.press("f4")
        #time.sleep(1)
        pyautogui.keyUp("alt")
        Speak("Tamam efendim")

    elif "pencereyi küçült" in query:
        pyautogui.keyDown("win")
        pyautogui.press("down")
        #time.sleep(1)
        pyautogui.keyUp("win")
        Speak("Tamam efendim")

    elif "pencereleri küçült" in query:
        pyautogui.keyDown("win")
        pyautogui.press("m")
        #time.sleep(1)
        pyautogui.keyUp("win")
        Speak("Tamam efendim")

    elif "pencereyi büyüt" in query:
        pyautogui.keyDown("win")
        pyautogui.press("up")
        #time.sleep(1)
        pyautogui.keyUp("win")
        Speak("Tamam efendim")

    elif "tuşuna bas" in query:
        button = query.replace(" tuşuna bas", "")
        button = button.replace("jarvis ", "")
        pyautogui.press(button)
        Speak("Tamam efendim")

    #####################################################################

    elif "play listemi aç" in query:
        webbrowser.open("https://www.youtube.com/watch?v=H9aq3Wj1zsg&list=RDH9aq3Wj1zsg&start_radio=1")
        Speak("Playlistiniz açılıyor efendim")

    elif "hız testi yap" in query or "internet testi yap" in query or "wifi testi yap" in query:
        Speak("Tamam efendim 10 15 saniye bekleyiniz")
        speed = speedtest.Speedtest()
        download = speed.download()
        upload = speed.upload()
        correctDown = int(download/800000)
        correctUp = int(upload/800000)
        Speak(f"İndirme hızı {correctDown-10} mbps ve yükleme hızı {correctUp-10} mbps")

    elif "ekran resmi al" in query or "ss al" in query:
        img= pyautogui.screenshot()
        home_directory = os.path.expanduser( '~' )
        img.save(f"{home_directory}/Desktop/screenshot.png")
        Speak("Ekran resmi alındı efendim")

    ################################################################################

    elif "tarayıcıyı kapat" in query:
        os.system("taskkill /f /im msedge.exe")
        Speak("Tarayıcı kapatılıyor efendim")

    elif "bilgisayarı kapat"  in query:
        Speak("Tamam efendim bilgisayar kapatılıyor")
        os.system("shutdown /s /t 5")

    elif "bilgisayarı yeniden başlat" in query:
        Speak("Tamam efendim bilgisayar yeniden başlatılıyor")
        os.system("shutdown /r /t 5")

    elif "bilgisayarı uyku moduna al" in query or "bilgisayarı uyut" in query:
        Speak("Tamam efendim bilgisayar uyku moduna alınıyor")
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

    elif "her şeyi uyku moduna al" in query or "her şeyi uyut" in query:
        Speak("Tamam efendim herşey uyku moduna alınıyor")
        requests.get(f"{wled_IP_Addr}/win&T=0")
        # arduino.write(b'1')
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

    ###################################################################################

    elif "hahaha" in query or "he he" in query:
        playsound("./SoundEffects/laugh.mp3")

    elif "osur" in query or "osuruk sesi" in query or "gaz çıkart" in query or "gaz çıkar" in query:
        farts = random.choice(["./SoundEffects/fart.mp3","./SoundEffects/fart2.mp3"])
        playsound(farts)

    # elif "şifreyi kır" in query:
    #     Speak("Tamam efendim şifre kırma modulu çalıştırılıyor")
    #     def PasswordHac():
    #         PasswordHack.toggle_fullscreen()
    #         PasswordHack.play()
    #         time.sleep(28)
    #         PasswordHack.stop()
    #         #Speak("şifre başarıyla kırıldı efendim")
    #     t1 = threading.Thread(target=PasswordHac)
    #     t1.start()

    # elif "kendi arayüzünü aç" in query:
    #     Speak("Tamam efendim")
    #     def JarvisU():
    #         JarvisUI.toggle_fullscreen()
    #         JarvisUI.play()
    #         time.sleep(62)
    #         JarvisUI.stop()
    #     t1 = threading.Thread(target=JarvisU)
    #     t1.start()

    # elif "kendi arayüzünü kapat" in query:
    #     JarvisUI.stop()
    #     Speak("Tamam efendim")

    elif "robota bağlan" in query:
        Speak("Robota geçiliyor efendim. Hazır efendim")
        while True:
            try:
                command = takecommand()
                if "ileri git" in command:
                    getRequest = requests.get(f"{robot_IP_Addr}/?State=F")
                    time.sleep(0.3)
                    getRequest = requests.get(f"{robot_IP_Addr}/?State=S")
                elif "geri git" in command:
                    getRequest = requests.get(f"{robot_IP_Addr}/?State=B")
                    time.sleep(0.3)
                    getRequest = requests.get(f"{robot_IP_Addr}/?State=S")
                elif "sağa dön" in command:
                    getRequest = requests.get(f"{robot_IP_Addr}/?State=R")
                    time.sleep(0.3)
                    getRequest = requests.get(f"{robot_IP_Addr}/?State=S")
                elif "sola dön" in command:
                    getRequest = requests.get(f"{robot_IP_Addr}/?State=L")
                    time.sleep(0.3)
                    getRequest = requests.get(f"{robot_IP_Addr}/?State=S")
                elif "çık" in command:
                    Speak("Tamam efendim robottan çıkılıyor")
                    break

                if getRequest:
                    getRequest_Status_Code = getRequest.status_code
                    print(f"Get Request status: {getRequest_Status_Code}")
                    if (getRequest_Status_Code>=200 and getRequest_Status_Code<299):
                        Speak("Hazır efendim")
                    else:
                        Speak("Bir hata oluştu efendim")
                    getRequest=0;

            except:
                Speak("Bir hata oluştu efendim")

    elif "kameraya bağlan" in query:
        Speak("Kameraya geçiliyor efendim. Hazır efendim")
        while True:
            try:
                command = takecommand()
                if "sağa çevir" in command:
                    getRequest = requests.get(f"{camera_IP_Addr}/cgi-bin/action?action=cam_mv&diretion=cam_right&lang=eng")
                    getRequest = requests.get(f"{camera_IP_Addr}/cgi-bin/action?action=cam_mv&diretion=cam_right&lang=eng")
                    getRequest = requests.get(f"{camera_IP_Addr}/cgi-bin/action?action=cam_mv&diretion=cam_right&lang=eng")
                elif "sola çevir" in command:
                    getRequest = requests.get(f"{camera_IP_Addr}/cgi-bin/action?action=cam_mv&diretion=cam_left&lang=eng")
                    getRequest = requests.get(f"{camera_IP_Addr}/cgi-bin/action?action=cam_mv&diretion=cam_left&lang=eng")
                    getRequest = requests.get(f"{camera_IP_Addr}/cgi-bin/action?action=cam_mv&diretion=cam_left&lang=eng")
                elif "yukarı çevir" in command or "yukarıya çevir" in command:
                    getRequest = requests.get(f"{camera_IP_Addr}/cgi-bin/action?action=cam_mv&diretion=cam_up&lang=eng")
                elif "aşağı çevir" in command or "aşağıya çevir" in command:
                    getRequest = requests.get(f"{camera_IP_Addr}/cgi-bin/action?action=cam_mv&diretion=cam_down&lang=eng")
                elif "çık" in command:
                    Speak("Tamam efendim kameradan çıkılıyor")
                    break

                if getRequest:
                    getRequest_Status_Code = getRequest.status_code
                    print(f"Get Request status: {getRequest_Status_Code}")
                    if (getRequest_Status_Code>=200 and getRequest_Status_Code<299):
                        Speak("Hazır efendim")
                    else:
                        Speak("Bir hata oluştu efendim")
                    getRequest=0;

            except:
                Speak("Bir hata oluştu efendim")

    elif "köpeğe bağlan" in query:
        Speak("Köpeğe bağlanılıyor efendim. Hazır efendim")
        while True:
            try:
                command = takecommand()
                if "ayağa kalk" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/standUp")
                elif "yat" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/sleep")
                elif "otur" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/sit")
                elif "öne eğil" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/leanForward")
                elif "sağa eğil" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/leanRight")
                elif "sola eğil" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/leanLeft")
                elif "el salla" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/wavingHand")
                elif "ileri git" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walk")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walk")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walk")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walk")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/standUp")
                elif "geri git" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkBack")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkBack")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkBack")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkBack")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/standUp")
                elif "sağa dön" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/turnRight")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/turnRight")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/turnRight")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/turnRight")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/standUp")
                elif "sola dön" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/turnLeft")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/turnLeft")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/turnLeft")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/turnLeft")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/standUp")
                elif "sağa git" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkRight")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkRight")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkRight")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkRight")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/standUp")
                elif "sola git" in command:
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkLeft")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkLeft")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkLeft")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/walkLeft")
                    getRequest = requests.get(f"{robotDog_IP_Addr}/standUp")
                elif "çık" in command:
                    Speak("Tamam efendim köpekten çıkılıyor")
                    break

                if getRequest:
                    getRequest_Status_Code = getRequest.status_code
                    print(f"Get Request status: {getRequest_Status_Code}")
                    if (getRequest_Status_Code>=200 and getRequest_Status_Code<299):
                        Speak("Hazır efendim")
                    else:
                        Speak("Bir hata oluştu efendim")
                    getRequest=0;

            except:
                Speak("Bir hata oluştu efendim")

checkArduino()
#os.startfile("E:\Wallpaper Engine\wallpaper64.exe")
os.startfile(".\Required\Rainmeter\Rainmeter.exe")
greeting()
while True:
    query = takecommand()
    if query !="":
        respond()
    # if query == "преминете към изкуствен интелект":
    #     Speak("Преминаване към изкуствен интелект", True)
    #     while True:
    #         query = takecommand()
    #         if query:
    #             if query == "çıkış yap":
    #                 Speak("çıkış yapılıyor", True)
    #                 break
    #             else:
    #                 openAI(query)

    if query == "чао" or query == "засппивай" or query == "довиждане":
        Speak("Добре, сър, можете да ми се обадите, ако имате нужда от мен", True)
        os.system("taskkill /f /im Rainmeter.exe")
        exit()