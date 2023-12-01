import pyautogui as py
import time
import clipboard
import cv2
import keyboard
import openai
from datetime import datetime
import csv
from datetime import datetime
import pandas as pd
import os
import telebot
import json
import sys



originalerUrl = ""


running = True

timestop = 0.2
timestopAktualisieren = 0.5



script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
dateipfad = os.path.join(script_dir, "Informationen für den Bot", "persönliche Daten.json")
with open(dateipfad, 'r') as json_datei:
        eigeneDaten = json.load(json_datei)

dateipfad = os.path.join(script_dir, "Informationen für den Bot", "Keys.json")
with open(dateipfad, 'r') as json_datei:
        keys = json.load(json_datei)




#das abschicken muss noch aktiviert werden, wenn die Testphase fertig ist

#was Ihr braucht:

#Bildschirmauflösung: 2560x1440
#Skalierung: 150%
#Opera Browser
#Immoscout24 PREMIUM ACCOUNT (wichtig!)
#zoom in Opera auf 100% #   strg/ctrl + mausrad     nach oben oder unten
#Opera in Dark Mode
#Tauschwohnungen deaktivieren #dazu wurde nichts programmiert
#adblocker aktivieren
#keine anderen Tabs offen haben
#keine anderen Programme offen haben
#keine anderen Fenster offen haben
#keine anderen Monitore haben
#keine anderen Browser offen haben
#keine anderen Programme im Hintergrund laufen lassen




#budget von mindestens 5€ für die Berliner // ca 1 cent kosten 5-7 Bewerbungen über ChatGPT je nach Größe des Anschreibens (die Größes der Ausgabe von ChatGPT ist unter keinen Umständen zu empfehlen, solange ChatGPT 5.0 noch nicht existiert)
#um mehr über die Kosten zu erfahren, gehe auf diese Seite: #https://gptforwork.com/guides/openai-gpt3-models

#Beenden des Scriptes mit ctrl+alt+shift

#Video schauen oder mich persönlich fragen, falls etwas ungenau sein sollte


#eigene Daten

premiumOrNot = str(eigeneDaten["premium"]).lower() #yes or no   #better of just get yourself a premium account and I'm not kidding.

Your_API_KEY_Tele = keys["Your_API_KEY_Tele"] #euer API Key von Telegram #https://core.telegram.org/bots/api#authorizing-your-bot
API_KEY = keys["API_KEY"] #euer API Key von OpenAI #https://platform.openai.com/account/api-keys
MODEL_NAME = keys["MODEL_NAME"] #welches ChatGPT modell soll verwendet werden? bitte nicht ändern, wenn ihr keine Ahnung dazu habt #https://gptforwork.com/guides/openai-gpt3-models 


#DATEN ERSETZEN MIT JSON DATEIEN





###################################################################################################################################################################################################################################
#AB DIESEN PUNKT NUR ÄNDERUNGEN VORNEHMEN, WENN IHR WISST, WAS IHR MACHT!!!!



def greatReset(id):
    #if active
    keyboard.press_and_release("ctrl+l")
    time.sleep(timestop)
    clipboard.copy(originalerUrl)
    time.sleep(timestop)
    keyboard.press_and_release("ctrl+v")
    time.sleep(timestop)
    keyboard.press_and_release("enter")
    main(id)


def überprüfungAufRoboter(id, indetificationSiteRoboter):
    if running == True:    
        time.sleep(1)
        max_loc, max_val = lookingForPixel("Roboter")
        if max_val > 0.90:
            print("Roboter erkannt")
            bot.send_message(id, "There is a captcha on the page, please solve it and then write 'done'")
            bot.register_next_step_handler_by_chat_id(id, check_captcha, indetificationSiteRoboter)
            print("5 sekunden vor exit")
            time.sleep(5)
            print("vor exit")
            return False
        else:
            print("Roboter nicht erkannt")
            return True

    else:
        print("es wurde \"beenden\" geschrieben")

def main(id):
    if running == True:
        while True:
            for i in range(12): #12 for not triggering the bot detection (the higher the better) 
                time.sleep(1)
                beenden() 
            
            
            lastStatus = aktualisierung(0, id)
            if lastStatus == False:
                break

            status = checkHearth()#checkt, ob das erste Herz rot oder weiss ist OBEN


            if status == True:
                print("erstes Herz ist weiß und es wurde die Seite geöffnet")
                statusTyp= schreibenDerBewerbung(id)
                if statusTyp == False:
                    print("die Seite wurde deaktiviert oder nachricht kann nicht abgeschickt werden")
                    keyboard.press_and_release("alt+left")
                elif statusTyp == "deaktivieren":
                    break
            else:
                pass
    else:
        print("es wurde \"beenden\" geschrieben")
openai.api_key = API_KEY



filePath_pictures = os.path.dirname(os.path.realpath(sys.argv[0]))+"\\"



def MouseMover():
    if running == True:
        py.moveTo(3, 3)
    else:
        print("es wurde \"beenden\" geschrieben")
def beenden2():
    if running == True:
       exit()
    else:
        print("es wurde \"beenden\" geschrieben")
def beenden():
    if running == True:
        if keyboard.is_pressed("ctrl+alt+shift"):
            exit()
    else:
        print("es wurde \"beenden\" geschrieben")

def lookingForPixel(name):
    if running == True:
        beenden()
        MouseMover()
        print("looking for " + name)
        time.sleep(0.2)
        picture1 = py.screenshot() #region=(1446, 597, 220, 230) minimap location  #start while in game

        picture1.save(os.path.join(filePath_pictures, "whatsapp.jpg"))

        TestJungle = cv2.imread(os.path.join(filePath_pictures, "Bilder\\", name + ".PNG"), cv2.IMREAD_UNCHANGED)
        DesktopTest = cv2.imread(os.path.join(filePath_pictures, "whatsapp.jpg"), cv2.IMREAD_UNCHANGED)

        nesne = cv2.cvtColor(DesktopTest, cv2.COLOR_BGR2GRAY)
        nesne1 = cv2.cvtColor(TestJungle, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(nesne1, nesne, cv2.TM_CCOEFF_NORMED)

        beenden()

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print("done looking for " + name)
        return max_loc, max_val
    else:
        print("es wurde \"beenden\" geschrieben")


def siteIsLoading(id, indetificationSite):
    if running == True:    
        max_loc, max_val = lookingForPixel("nichtAktualisiert")
        while max_val > 0.85:
            max_loc, max_val = lookingForPixel("nichtAktualisiert")
            print(max_val)
            time.sleep(timestopAktualisieren)
            beenden()
        print("die Seite ist fertig geladen")
        print("Untersuchung nach Anti-Bot")
        newStatus = überprüfungAufRoboter(id, indetificationSite)
        return newStatus
    else:
        print("es wurde \"beenden\" geschrieben")     


###############
def aktualisierung(identification, id):
    if running == True:
        max_loc, max_val= lookingForPixel("Aktualisieren")

        while max_val < 0.80:
            
                max_loc, max_val= lookingForPixel("Aktualisieren")
                print(max_val)
                time.sleep(timestopAktualisieren)
                beenden()
        py.moveTo(max_loc[0], max_loc[1])
        time.sleep(timestop)
        py.click()
        nextNewStatus = siteIsLoading(id, identification)
        return nextNewStatus
    else:
        print("es wurde \"beenden\" geschrieben")   
##############
def zumScrollen():
    if running == True:
        max_loc, max_val = lookingForPixel("zumScrollen")
        while max_val < 0.80:
            max_loc, max_val = lookingForPixel("zumScrollen")
            print(max_val)
            time.sleep(0.5)
            beenden()
        py.moveTo(max_loc[0], max_loc[1])
        time.sleep(timestop)
        py.click()
        time.sleep(timestop)
        py.scroll(-500)
    else:
        print("es wurde \"beenden\" geschrieben")
##########
def premiumOrNot1(Info):
    if running == True:
        if Info == "no":
            zumScrollen()
            time.sleep(timestop)
            keyboard.press_and_release("ctrl+l")
            time.sleep(timestop)
            keyboard.press_and_release("ctrl+c")
            time.sleep(timestop)
            originalUrl = clipboard.paste()
            return originalUrl
            
        else:
            if running == True:
                max_loc, max_val = lookingForPixel("zumScrollen")
                while max_val < 0.80:
                    max_loc, max_val = lookingForPixel("zumScrollen")
                    print(max_val)
                    time.sleep(0.5)
                    beenden()
                py.moveTo(max_loc[0], max_loc[1])
                time.sleep(timestop)
                py.click()
                time.sleep(timestop)
                keyboard.press_and_release("ctrl+l")
                time.sleep(timestop)
                keyboard.press_and_release("ctrl+c")
                time.sleep(timestop)
                originalUrl = clipboard.paste()
                return originalUrl
    else:
        print("es wurde \"beenden\" geschrieben")        
    
def save_variables_to_csv(var1, var2, csv_path='variables_db.csv'):
    if running == True:
        try:
            now = datetime.now()
            timeString = now.strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([timeString, var1, var2])

        except Exception as e:
            with open('error_log.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([timeString, str(e)])
    else:
        print("es wurde \"beenden\" geschrieben")

def read_and_sort_csv(csv_path='variables_db.csv'):
    if running == True:
        df = pd.read_csv(csv_path, names=['time', 'ChatGPTOutput', 'NameAdressETC'], encoding='ISO-8859-1')
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values(by=['time'], inplace=True, ascending=False)
        return df
    else:
        print("es wurde \"beenden\" geschrieben")


############
def checkHearth():
    if running == True:
        time.sleep(0.5) #um sicher zu gehen, dass die Seite 100% geladen wurde                  #URSPRÜNGLICH 1
        max_loc, max_val = lookingForPixel("oberesHerzWeiss")
        if max_val > 0.80:
            py.moveTo(max_loc[0]-420, max_loc[1]+110)
            print(max_val)
            time.sleep(timestop)
            py.click()
            return True
        else:
            return False
    else:
        print("es wurde \"beenden\" geschrieben")


def checkSite():
    if running == True:
        max_loc, max_val = lookingForPixel("notActive")
        if max_val > 0.80:
            return False
    else:
        print("es wurde \"beenden\" geschrieben")


def checkIfNoPicture():
    if running == True:
        max_loc, max_val = lookingForPixel("KeinBild")
        if max_val > 0.80:
            return True
        else:
            return False
    else:
        print("es wurde \"beenden\" geschrieben")
def fullAdressOrNot(adressSecond):
    if running == True:

        if adressSecond.strip() ==  "Die vollständige Adresse der Immobilie erhalten Sie vom Anbieter.":
            print("die vollständige Adresse ist nicht vorhanden, es wird nur die Stadt und die Postleitzahl provided")
            return False #die vollständige Adresse ist nicht vorhanden, es wird nur die Stadt und die Postleitzahl angegeben
        else:
            print("die vollständige Adresse ist vorhanden")
            return True #die vollständige Adresse ist vorhanden
    else:
        print("es wurde \"beenden\" geschrieben")



def findingAdress(id):
    if running == True:
        max_loc, max_val = lookingForPixel("Pin")
        counter = 0
        while max_val < 0.80 and counter <= 10:
            max_loc, max_val = lookingForPixel("Pin")
            print(max_val)
            time.sleep(0.5)
            #counter += 1       #wurde erstmal deaktiviert, da es zu Problemen führen kann
        if counter >= 10:
            greatReset(id)
        x, y = max_loc
        py.moveTo(x+200, y)
        for i in range(3):
            py.click()
        keyboard.press_and_release("ctrl+c")
        time.sleep(0.5)
        adressFirst = clipboard.paste()
        py.moveTo(x+400, y)
        for i in range(3):
            py.click()
        keyboard.press_and_release("ctrl+c")
        time.sleep(0.5)
        adressSecond = clipboard.paste()

        return adressFirst, adressSecond
    else:
        print("es wurde \"beenden\" geschrieben")
    #prompt = f"""Bitte verfassen Sie mir eine umwerfende Bewerbung um eine Wohnung. Sollten Informationen wie der zweite Teil der Adresse fehlen, sollte die Bewerbung so formuliert sein, dass sie immer noch mit den verfügbaren Daten arbeitet. Beginnen Sie die Bewerbung mit \"Sehr geehrte...\" oder \"Sehr geehrter...\" und verwenden Sie die gegebenen Informationen, die mir zur Verfügung stehen: Stadt und Postleitzahl der Adresse, Straßenname und Hausnummer (falls vorhanden) und den Namen oder die Firma des Vermieters oder der Kontaktperson. Informationen, die nicht verfügbar sind, sollten in der Bewerbung nicht durch Platzhalter wie '[]' repräsentiert werden.:   {text}"""

def gpt3(text):
    if running == True:
        prompt = f"""Bitte verfassen Sie mir eine umwerfende Bewerbung um eine Wohnung. Sollten Informationen wie der zweite Teil der Adresse fehlen, sollte die Bewerbung so formuliert sein, dass sie immer noch mit den verfügbaren Daten arbeitet. Beginnen Sie die Bewerbung mit \"Sehr geehrte...\" oder \"Sehr geehrter...\" und verwenden Sie die gegebenen Informationen, die mir zur Verfügung stehen: Stadt und Postleitzahl der Adresse, Straßenname und Hausnummer (falls vorhanden) und den Namen oder die Firma des Vermieters oder der Kontaktperson. Informationen, die nicht verfügbar sind, sollten in der Bewerbung nicht durch Platzhalter wie '[]' repräsentiert werden.:   {text}"""

        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.1,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            generated_text = response['choices'][0]['text'].strip()
            
                
            first_word = generated_text.split()[0]
            if str(first_word).lower() == "sehr":
                return generated_text
            else:
                filePathPicturesErrorReport = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\variables_db.csv"
                save_variables_to_csv(generated_text, prompt, filePathPicturesErrorReport)#1zeit #2ChatGPTOutput #3NameAdressETC
                raise Exception("The generated text does not meet   the requirements")
                    
        except Exception as e:
            print(f"Error occurred: {e}")
            return None
    else:
        print("es wurde \"beenden\" geschrieben")




def writingMessageViaGPT(ersteAdresse, zweiteAdresse, nameOrCompany, id):
    if running == True:
        if nameOrCompany.strip() == "Nachricht":
            nameOrCompany = "kein Name oder Firma vorhanden vorhanden"

        print(ersteAdresse, zweiteAdresse, nameOrCompany)

        textZumBearbeiten = f"""
        \"Die vollständige Adresse oder nur ein Teil der Adresse, worauf wir uns bewerben möchten: \"{ersteAdresse + zweiteAdresse}\",
        Name des Ansprechpartners/Kontaktperson oder Unternehmens, welche das Objekt vermietet: \"{nameOrCompany}\",
        Mein Name (Bewerber): \"{eigeneDaten["name"]}\",
        Meine E-Mail-Adresse: \"{eigeneDaten["email"]}\",
        Meine Telefonnummer: \"{eigeneDaten["telefon"]}\",
        Meine Adresse: \"{eigeneDaten["adresse"]}\",
        Meine Postleitzahl: \"{eigeneDaten["plz"]}\",
        Meine Stadt: \"{eigeneDaten["stadt"]}\",
        Mein Land: \"{eigeneDaten["land"]}\",
        Mein Alter: \"{eigeneDaten["alter"]}\",
        Mein Geburtsort: \"{eigeneDaten["geburtsort"]}\",
        Mein Geburtsland: \"{eigeneDaten["geburtsland"]}\",
        Mein Familienstand: \"{eigeneDaten["familienstand"]}\",
        Meine Nationalität: \"{eigeneDaten["nationalität"]}\",
        Mein Beruf: \"{eigeneDaten["beruf"]}\",
        Mein Arbeitgeber oder meine Schule: \"{eigeneDaten["arbeitgeberOderSchule"]}\",
        Meine Haustiere: \"{eigeneDaten["haustiere"]}\",
        Ich bin Raucher: \"{eigeneDaten["raucher"]}\"\"
        """ 
        getText = gpt3(textZumBearbeiten)
        if getText == None or getText == "":
            print("Fehler beim Schreiben der Nachricht, da der Inhalt leer ist")
            print("Immobilie wird gespeichert als \"angeschrieben\" und das Ergebnis kommt in eine Log.txt")
            max_loc, max_val = lookingForPixel("merken")
            while max_val < 0.80:
                max_loc, max_val = lookingForPixel("merken")
            py.moveTo(max_loc)
            time.sleep(timestop)
            py.click()
            time.sleep(5)
            keyboard.press_and_release("alt+left")
            newStatusGPT = aktualisierung(0, id)
            if newStatusGPT == False:
                return False
            main(id)        #muss geändert werden. lieber das script neu starten, als die function neu aufzurufen

        else:
            return getText
    else:
        print("es wurde \"beenden\" geschrieben")    




def openingMessage(id):
    if running == True:
        max_loc, max_val = lookingForPixel("ButtonNachricht")
        counter = 0
        while max_val < 0.80 and counter <= 10:
            max_loc, max_val = lookingForPixel("ButtonNachricht")
            print(max_val)
            time.sleep(0.5)
            #counter += 1       #wurde erstmal deaktiviert, da es zu Problemen führen kann
        if counter >= 10:
            greatReset(id)
        x, y = max_loc
        py.moveTo(x+40, y+40)
        time.sleep(timestop)
        py.click()
    else:
        print("es wurde \"beenden\" geschrieben")

def schreibenDerNachricht(newText, id):
    if running == True:
        
        max_loc, max_val = lookingForPixel("IhreNachricht")
        count = 0
        while max_val < 0.80 and count <= 7:
            max_loc, max_val = lookingForPixel("IhreNachricht")
            print(max_val)
            count += 1
            time.sleep(1)
        if count >= 7:
            print("es scheint so, dass der Button deaktiviert ist, like wird vergeben und die Seite wird ignoriert")
            max_loc, max_val = lookingForPixel("schliessenPlus")
            if max_val > 0.90:
                py.click(max_loc)
            return False
        
        else:
            py.moveTo(max_loc[0]+100, max_loc[1]+100)
            time.sleep(timestop)
            py.click()
            time.sleep(timestop)
            keyboard.press_and_release("ctrl+a")
            time.sleep(timestop)
            clipboard.copy(newText)
            time.sleep(timestop)
            keyboard.press_and_release("ctrl+v")
            time.sleep(timestop)
            py.click(max_loc[0], max_loc[1])
            time.sleep(timestop)
            py.scroll(-1500)
            time.sleep(timestop)
            counter = 0
            max_loc, max_val = lookingForPixel("absender")
            while max_val < 0.80 and counter <= 10:
                max_loc, max_val = lookingForPixel("absender")
                print(max_val)
                time.sleep(0.5)
                #counter += 1       #wurde erstmal deaktiviert, da es zu Problemen führen kann
            if counter >= 10:
                greatReset(id)
            py.moveTo(max_loc[0], max_loc[1])
            time.sleep(timestop)
            py.click() #clickt auf den Absender
            for i in range(8):
                time.sleep(1)
                beenden()
            keyboard.press_and_release("alt+left")
            return True
    else:
        print("es wurde \"beenden\" geschrieben")



def antoherCheckForAvail():
    if running == True:
        max_loc, max_val = lookingForPixel("merken")
        time.sleep(timestop)
        py.moveTo(max_loc[0], max_loc[1])
        time.sleep(timestop)
        py.click()
        time.sleep(timestop)
        return False
    else:
        print("es wurde \"beenden\" geschrieben")


def schreibenDerBewerbung(id):
    if running == True:
        
        
        #sicher gehen, dass die Seite funktioniert
        nextNewStatusBewerbung = siteIsLoading(id, 1) #sicher gehen, dass die Seite geladen wurde und alle Elemente mit dem menschlichen Auge ersichtlich sind
        if nextNewStatusBewerbung == False:
            return "deaktivieren"
        statusTyp = checkSite() #checkt, ob die Seite aktiv ist oder deaktiviert wurde
        if statusTyp == False:
            return False




        max_loc, max_val = lookingForPixel("ButtonNachricht")
        while max_val < 0.80:
            max_loc, max_val = lookingForPixel("ButtonNachricht")    
        x, y = max_loc

        statusTypSecond= checkIfNoPicture() #checkt, ob ein Bild vorhanden ist, um die Position des Textes zu ermitteln
    
    #2026 911,
    
        if statusTypSecond == True:
            print("Status Typ ist KEINBILD")
            py.moveTo(x+500, y-78)
            print(x+500, y-90)
            time.sleep(timestop)
            for i in range(3):
                py.click()
            keyboard.press_and_release("ctrl+c")
            time.sleep(0.5)
            nameOrCompany = clipboard.paste()
            py.scroll(-500)
            ersteAdresse, zweiteAdresse = findingAdress(id) #if Status == False, dann ist die Adresse nicht vorhanden



        elif statusTypSecond == False:
            print("Status Typ ist BILD")
            py.moveTo(x+500, y-130)
            time.sleep(timestop)
            for i in range(3):
                py.click()
            keyboard.press_and_release("ctrl+c")
            time.sleep(0.5)
            nameOrCompany = clipboard.paste()
            py.scroll(-500)
            ersteAdresse, zweiteAdresse = findingAdress(id) #if Status == False, dann ist die Adresse nicht vorhanden


        newText = writingMessageViaGPT(ersteAdresse, zweiteAdresse, nameOrCompany, id)
        if newText == False:
            return False
        py.scroll(500)
        openingMessage(id)
        time.sleep(timestop)
        statusDeak= schreibenDerNachricht(newText,id)
        if statusDeak == False:
            antoherCheckForAvail()
            keyboard.press_and_release("alt+left")
            

        else:
            pass
    else:
        print("es wurde \"beenden\" geschrieben")



bot = telebot.TeleBot(token=Your_API_KEY_Tele)
test = "Start"
@bot.message_handler(func=lambda message: "Start" in message.text or "start" in message.text or "START" in message.text or "beenden" in message.text)
def greet(message):
    print("es funktioniert")
            

    if message.text == "beenden":
        bot.send_message(message.chat.id, "Damit hast du nun den Bot beendet.")
        global running
        running = False

    

    if message.text == "Start" or message.text == "start" or message.text == "START":
        
        running = True
        bot.send_message(message.chat.id, "Damit hast du nun den Bot gestartet.")  
        global originalerUrl
        originalerUrl= premiumOrNot1(premiumOrNot) #wird rausgenommen, sobald es öffentlich ist, außer es wird auch für nicht premium user verfügbar sein
        
        main(message.chat.id)

def check_captcha(message, identification):
    print("telegram")
    time.sleep(5)
    print("telegram nochmal")
    if message.text == "done":
        # Captcha was solved, continue with the script
        bot.send_message(message.chat.id, "Script execution continued.")
        if identification == 0:
            main(message.chat.id)
        elif identification == 1:
            schreibenDerBewerbung(message.chat.id)


    else:
        # Captcha was not solved, check the next message again
        bot.send_message(message.chat.id, "Please solve the captcha and write 'done'")
        bot.register_next_step_handler(message, check_captcha)


    

bot.polling()
