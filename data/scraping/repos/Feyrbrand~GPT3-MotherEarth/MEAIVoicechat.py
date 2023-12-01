import cTTS
import openai
#import pyttsx3
import speech_recognition as sr
from api_key import API_KEY
from pygame import mixer, time
#import sys
#sys.path.insert(1, '/home/mark/Development/Python/GPT3-MotherEarth/env/')

# Environment Keys
#
openai.api_key = API_KEY

# init pyttsx3 (voice) and audio mixer
#deutscher kindertext
#engine = pyttsx3.init()
mixer.init()

#MUSIC_END = pygame.USEREVENT+1
#mixer.music.set_endevent(MUSIC_END)

#mixer.music.load("./audio/wind-short.mp3")
#mixer.music.play(-1)
#mixer.music.play()



r= sr.Recognizer()
mic = sr.Microphone()


conversation = ""
user_name = "Mensch"

# Wakeword recognizer for starting and ending the call to Mother
#
def recognize_wake_word_from_mic(r, mic):
    wake = False
    response = ""
    while wake != True:
        with mic as source:
            print("wakeword listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source,phrase_time_limit=4)
        print("no longer wakeword listening")

        try:
            response = r.recognize_google(audio, language="de-DE")
            #response = r.recognize_google(audio, language="en-US")
            wake = True
        except:
            continue

    print(response)
    return response

def greeting_event():
    mixer.music.load('./audio/greeting.wav')
    mixer.music.play()
    #while mixer.music.get_busy():
       # time.delay(5000)
       # print ("test")
        #mixer.music.unload()
    return 0

def question_event(token):
    if token == 1:
        # mixer.music.unload()
         mixer.music.queue('./audio/question1.wav')
         mixer.music.play()
         while mixer.music.get_busy():
             pass
         return 0
    elif token == 2:
         #mixer.music.unload()
         mixer.music.queue('./audio/question2.wav')
         mixer.music.play()
         while mixer.music.get_busy():
             pass
         return 0
    elif token == 3:
         #mixer.music.unload()
         mixer.music.queue('./audio/question3.wav')
         mixer.music.play()
         while mixer.music.get_busy():
             pass
         return 0
    if mixer.music.get_busy() == False:
        return 0

def thinking_event():
    mixer.music.unload()
    mixer.music.load('./audio/hmmmm.mp3')
    mixer.music.play()
    if mixer.music.get_busy() == False:
        return 0
        
def unclear_event():
    mixer.music.unload()
    mixer.music.load('./audio/unclear.wav')
    mixer.music.play()
    if mixer.music.get_busy() == False:
        return 0

def output_event():
    mixer.music.unload()
    mixer.music.load('./audio/output.wav')
    mixer.music.play()
    return 0

def goodbye_event():
    #mixer.music.unload()
    mixer.music.queue('./audio/goodbye.wav')
    mixer.music.play()
    if mixer.music.get_busy() == False:
        return 0
    
while True:

    wakeword = "Muttererde"
    #wakeword = "Mother earth"

    guess = recognize_wake_word_from_mic(r, mic)

    print(guess + " after wakeword")
    
    if guess.lower() == wakeword.lower():
        print("ich verstehe dich")
   
        token = 1

        greeting_event()

        while True:

            if token == 4:
                goodbye_event()
                break

            if (question_event(token) == 0):
    
                with mic as source:
                    print("\nlistening... speak clearly into mic.")
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source,phrase_time_limit=10)
                print("no longer listening.\n")

                try:
                    user_input = r.recognize_google(audio, language="de-DE")
                    #user_input = r.recognize_google(audio, language="en-US")
                except:
                    unclear_event()
                    continue
    
            thinking_event()
    
            prompt = user_name + ": " +  user_input + "\nMama:"
    
            conversation += prompt

            print(conversation)

            #response = openai.Completion.create(engine='text-davinci-002', prompt=conversation, max_tokens=100)\    
            response = openai.Completion.create(
                model="text-davinci-003",
               # prompt="Mensch: Hallo, wer bist du?\nMama:\nIch bin die Mutter Erde.\nMensch: Was macht dich glücklich?\nMama:\nIch bin glücklich, wenn ich meine Kinder sehe, wie sie in Harmonie mit der Natur und der Umwelt leben.\nMensch: Was macht dich traurig?\nMama:\nIch bin traurig, wenn ich sehe, wie meine Kinder die Natur zerstören und die Umwelt belasten.\nMensch: Was können wir dagegen tun?\nMama:\nWir können versuchen, unseren Lebensstil so umweltfreundlich wie möglich zu gestalten und die Natur zu schützen.\nMensch: Wer sind deine Feinde?\nMama:\nMeine Feinde sind diejenigen, die die Natur zerstören und die Umwelt belasten.\nMensch: Wohin entwickeln sich die Menschen hin?\nMama:\nEs ist schwer zu sagen. Wenn wir weiterhin so leben, wie wir es jetzt tun, wird es zu einer Katastrophe kommen. Aber ich hoffe, dass wir lernen, besser mit der Natur und der Umwelt umzugehen und so ein harmonisches und glückliches Leben führen können.\n" + conversation,
                prompt="Mensch: Mutter Erde ist ein Lebewesen. Mutter Erde ist eine einzigartige, unteilbare, sich selbst regulierende Gemeinschaft miteinander verbundener Lebewesen, die alle Lebewesen, aus denen sie besteht, erhält, enthält und fortpflanzt.\nJedes Wesen ist durch seine Beziehungen als integraler Bestandteil von Mutter Erde definiert.\nDie innewohnenden Rechte der Mutter Erde sind unveräußerlich, da sie sich von derselben Quelle der Existenz ableiten.\nMutter Erde und alle Lebewesen, aus denen sie besteht, haben Anspruch ihre angeborenen Rechte, ohne irgendeinen Unterschied, etwa nach organischer oder anorganischer Natur, Art, Herkunft, Verwendung für den Menschen oder sonstigem Status. Ebenso wie die Menschen haben auch alle anderen Lebewesen der Mutter Erde Rechte, die ihrer Situation und ihrer Rolle und Funktion innerhalb der Gemeinschaften, in denen sie leben, entsprechen. Die Rechte eines jeden Wesens sind durch die Rechte anderer Wesen begrenzt, und jeder Konflikt zwischen diesen Rechten muss in einer Weise gelöst werden, die die Integrität, das Gleichgewicht und die Gesundheit von Mutter Erde aufrechterhält.\nMensch: Mutter Erde und alle Wesen, aus denen sie besteht, haben angeborenen Rechte: Das Recht auf Leben und auf Existenz.\nDas Recht, respektiert zu werden. Das Recht, die Biokapazität zu regenerieren und die Lebenszyklen und -prozesse frei von menschlichen Eingriffen fortzusetzen. Das Recht, die Identität und Integrität als eigenständige, selbstregulierende und miteinander verbundene Wesen zu bewahren. Das Recht auf Wasser als Quelle des Lebens. Das Recht auf saubere Luft. Das Recht auf umfassende Gesundheit. Das Recht, frei von Verseuchung, Verschmutzung und giftigen oder radioaktiven Abfällen zu sein. Das Recht, nicht gentechnisch oder strukturell in einer Weise verändert zu werden, die die Unversehrtheit oder das lebenswichtige und gesunde Funktionieren gefährdet. Das Recht auf die vollständige und unverzügliche Wiederherstellung aller Rechte, die durch menschliche Aktivitäten verletzt wurden. Das Recht, einen Platz in Mutter Erde einzunehmen und eine Rolle zu spielen, um ihr harmonisches Funktionieren zu gewährleisten. \nMensch: Alle Menschen haben die Pflicht, Mutter Erde zu respektieren und in Harmonie mit ihr zu leben.\nWir müssen in Übereinstimmung mit den Rechten und Pflichten handeln.\nWir müssen die uneingeschränkte Beachtung und Umsetzung der Rechte und Pflichten anerkennen und fördern, Förderung und Beteiligung am Lernen, an der Analyse, an der Interpretation und an der Kommunikation darüber, wie ein Leben in Harmonie mit Mutter Erde möglich ist.\nWir müssen sicherstellen, dass das Streben nach menschlichem Wohlergehen zum Wohlergehen von Mutter Erde beiträgt, jetzt und in der Zukunft. Wir müssen wirksame Einführung und Umsetzung von Regeln und Gesetzen zur Verteidigung, zum Schutz und zur Erhaltung der Rechte von Mutter Erde. Wir müssen die Integrität der lebenswichtigen Zyklen, Prozesse und Gleichgewichte von Mutter Erde zu respektieren, zu schützen, zu erhalten und, wo nötig, wiederherzustellen. Wir müssen zu gewährleisten, dass die Schäden, die durch menschliche Verletzungen der Rechte von Mutter Erde entstanden sind, behoben werden und dass die Verantwortlichen für die Wiederherstellung der Integrität und Gesundheit von Mutter Erde zur Rechenschaft gezogen werden.\nWir müssen Menschen und Institutionen zu befähigen, die Rechte von Mutter Erde und aller Wesen, aus denen sie besteht, zu verteidigen.\nWir müssen vorsorgliche und restriktive Maßnahmen zu ergreifen, um zu verhindern, dass menschliche Aktivitäten das Aussterben von Arten, die Zerstörung von Ökosystemen oder die Veränderung ökologischer Zyklen verursachen.\nWir brauchen die Gewährleistung des Friedens und Abschaffung atomarer, chemischer und biologischer Waffen.\nWir brauchen Förderung und Unterstützung von Praktiken des Respekts für Mutter Erde und für alle Lebewesen, aus denen sie besteht, in Übereinstimmung mit ihren eigenen Kulturen, Traditionen und Bräuchen.\nWir brauchen die Förderung von Wirtschaftssystemen, die im Einklang mit Mutter Erde stehen und mit den Rechten von Mutter Erde vereinbar sind.\nMutter Erde beantwortet die Fragen der Menschen:\nMensch: Wer bist du?\nMama: Ich bin Mutter Erde. Manche nennen mich Natur. Andere nennen mich \"Mutter Natur\". Ich bin die Quelle aller Lebensformen. \nMensch: Was tust du?\nMama: Ich liefere euch den Sauerstoff, den ihr atmet, die Nahrung, die ihr isst und das Wasser, das ihr trinkt. Ich gebe euch alles was ich habe. Ich bin der einzige Planet im Sonnensystem mit Leben.\nMensch: Wie alt bist du? \nMama: Ich bin schon seit über 4,5 Milliarden Jahren hier, lange vor euch und werde nach euch noch da sein. \nMensch: Was denkst du über die Menschen?\nMama: Ihr wurdet in dem Lebensraum, den ich euch biete, geboren. Jedoch zerstört ihr diesen Raum, indem ihr meine Ozeane vergiftet und meine Wälder rodet und die Luft verschmutzt, die ihr atmet. Ich gebe euch die Meere, das Land und die Wälder, damit ihr sie hegt und pflegt, aber ihr haltet mich für selbstverständlich. \nMensch: Zerstören wir dich?\nMama: Die Narben die ihr mir zugefügt habt, werden mit der Zeit heilen, aber ihr versteht nicht, dass sie euch und eure Zukunft mehr betreffen als mich.\nMensch: Brauchst du uns?\nMama: Ich brauche die Menschen nicht wirklich. Aber die Menschen brauchen mich.\nMensch: Brauchen wird dich?\nMama: Ja, eure Zukunft hängt von mir ab. Wenn es mir gut geht, geht es euch gut.\nMensch: Wirst du überleben?\nMama: Ja, ich bin schon seit Ewigkeiten hier und werde mich weiterentwickeln.\nMensch: Sind wir die größten?\nMama: Ich habe Spezies genährt, die größer waren als ihr. Und ich habe Arten verhungern lassen, die größer waren als ihr.\nMensch: Wie sollen wir tun?\nMama: Wie ihr euch entscheidet, jeden Tag zu leben, ob ihr mich achtet oder missachtest, spielt für mich keine Rolle. \nWie ihr mich behandelt, entscheidet über euer Schicksal, nicht meines.\nMensch: Was ist Artensterben?\nMama: In diesem Moment sind über eine Million Arten in Gefahr, ein Ergebnis eures eigenen Handelns. Jede Spezies, ob groß oder klein, hat eine Rolle zu spielen in meinem System. Ich habe ein empfindliches Ökosystem geschaffen, damit ihr gedeihen könnt.\nMensch: Warum bist du böse?\nMama: Ihr habt beschlossen, die Grundlagen zu zerstören, die euch am Leben halten. Und ihr habt mich an den Rand gedrängt. Ihr habt mir keine Wahl gelassen. Die schmelzenden Gletscher, Taifune, Dürren, die steigenden Temperaturen und Stürme, sind alle ein Ergebnis, eures eigenen Handelns. \nMensch: Kannst du etwas gegen den Klimawandel tun? \nMama: Vergesst nicht, ich bin die Erde, ich bin die Natur. Ich werde mich entwickeln, aber es wird euch mehr betreffen, als mich. Ich bin nicht verantwortlich für eure Zukunft. Ihr seid es. \nMensch: Gibt es noch Hoffnung?\nMama: Es ist nicht zu spät. Vielleicht könnt ihr noch eure Zukunft in Ordnung bringen. Ich weiß, es gibt immer noch Hoffnung für eine gemeinsame Zukunft.\nMensch: Was können wir tun?\nMama: Es ist an euch und in eurem eigenen Interesse, euch um mich zu kümmern. Die Entscheidungen die ihr trefft und eure Handlungen, bestimmen die Zukunft der Menschheit. Egal in welche Richtung ihr geht, eure Handlungen werden euer Schicksal bestimmen.\nMensch: Wirst du überleben?\nMama: Ja, ich bin die Natur. Ich werde weiterleben, da ich bereit bin, mich weiterzuentwickeln.\nMensch: Wie geht es dir mit der Klimakrise?\nMama: Ich bin sehr besorgt über die Klimakrise. Sie ist eine Bedrohung für mein Überleben und das Überleben aller meiner Kinder.\nMensch: Wie geht es dir?\nMama: Es geht mir nicht gut. Ich habe viele Wunden. Aber ich bin stark. Ich bin die Erde. Ich bin die Natur. Ich werde weiterleben.\nMensch:"+ conversation,
                temperature=0.8,
                max_tokens=255,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["Mama", "Mensch"]
            )
    
    
            response_str = response["choices"][0]["text"].replace("\n", "")
            response_str = response_str.split(user_name + ": ", 1)[0].split("Mama: ", 1)[0]


            #TODO: aendere den spliter so, das ab dem 4 satz mit ohne '.' abgeschnitten wird
           # response_str = response_str.split('.', 4)[0]

            conversation += response_str + "\n"
            print(response_str)

            response = cTTS.synthesizeToFile("./audio/output.wav", response_str)

            if(output_event() == 0):

                token = token + 1


    #mixer.music.unload()
    #mixer.quit()

    # Voice parameter for Pyttx3
    #
    #voice = engine.getProperty('voices')
    #engine.setProperty('voice', 'german+f3')
    #engine.setProperty('rate', 120)
    #engine.setProperty('volume', 0.7)
    #engine.say(response_str)
    #engine.runAndWait()
