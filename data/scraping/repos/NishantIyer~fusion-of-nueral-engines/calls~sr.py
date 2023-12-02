import speech_recognition as sr
import playsound  # to play saved mp3 file
from gtts import gTTS  # google text to speech
import os  # to save/open files
import openai
import requests
from PIL import Image
from io import BytesIO
num = 1
andh = "The allegations in the complaint are that the petitioner, who is the husband of the de facto complainant, had demanded Rs.1,00,000/- as dowry and when the same was not paid, the petitioner had treated the de facto complainant with cruelty and had also assaulted her. </p> This data summarizes four cases from Andhra Pradesh High Court - Amravati involving the State of Andhra Pradesh, M/S.Vyshno Constructions, Attunuri Koti Reddy A. Rama Koti, Kollu Ankababu, and Nimmagadda Rama Krishna. In the first case, the court is assessing whether or not the writ petitioners/contractors are entitled to interest in terms of clause 43 of the Conditions of Contract. In the second case, a complaint was filed against accused Nos. 1 to 6 for suspecting the deceased of theft which led to him committing suicide. The third case examines a journalist accused of forwarding a social media post which alleged the wife of a key-officer in the Chief Minister's office was involved in a gold smuggling investigation.  "
bom = "This data pertains to four different court cases in the Bombay High Court on December 2, 2022. In the first case, the petitioners, borrowers, sought a writ of mandamus and declarations that a notification issued under the Recovery of Debts and Bankruptcy Act, 1993 be declared unconstitutional, that the transfer of their case from one Debt Recovery Tribunal to another be declared void, and that the Micro, Small and Medium Enterprises Development Act, 2006 prevail over the Securitisation and Reconstruction of Financial Assets and Enforcement of Security Interest Act, 2002. In the second case, the appellants were convicted and sentenced to life imprisonment and a fine of Rs.5000/- for the offense punishable under Section 302 read with Section 34 of the Indian Penal Code. In the third case, the appellant, an automobile manufacturing company, was held to be entitled to recover an amount of Rs.12,80,480/- from the respondent. In the fourth case, the respondent alleged that after marriage she was maltreated by the applicants and driven out of the house after they demanded Rs.1,00,000/- for a house. "
madras = "Respondents This case is about M/S.Malur Tubes Private Limited vs The State Of Tamil Nadu on 2 December, 2022. The Petitioners are represented by Mr.A.Ramesh, Senior Counsel and Mr.C.Arunkumar while the Respondents are represented by Mr.S.Balaji, Government Advocate (Crl.Side). The Petitioners are Shri Lakshmi Metal Udyog Ltd., represented by its Director, Mr.Vinod Kumar Singhal, and Mr.Vinod Kumar Singhal himself, while the Respondents are the State of Tamil Nadu and the Inspector of Police, Central Crime Branch, Egmore, Chennai. The case is C.R.P.(MD)No.5438 of 2015. "
sup = "Resettlement Act, 2013 (hereinafter referred to as “Act, 2013”), the Government of NCT of Delhi and Anr. have preferred the present appeal. This data summarizes five appeals filed by the Land Acquisition Collector, Central Bureau of Investigation, State of Jharkhand and other authorities, Government of NCT of Delhi and original applicants - plaintiffs against the judgments and orders passed by the High Courts of Delhi, Madhya Pradesh, Kerala, Jharkhand and Madras respectively. The appeals are filed for acquisitions of lands and granting of anticipatory bail to the accused. "

beng = "on 6 December,  2022This data describes four cases being heard in the Calcutta High Cour on 6 December, 2022. In the first case, Remington Rand Of India Ltd vs Jaypee Trading Company Ltd, there are several lawyers representing the applicant. In the second case, Avlokan Commosales Private Ltd vs State Bank Of India & Anr, a settlement was reached between the borrower and Phoenix ARC and a sale certificate was issued. In the third case, Tripti Ranjan Roy vs Rajanvir Singh Kapur, the petitioner claimed for subsistence allowance and additional costs. In the fourth case, Dredging And Desiltation Company vs Mackintosh Burn And Northern, there are several lawyers representing the plaintiff and defendants. The fifth case, Principal Commissioner Of Income Tax vs Jis Foundation, requires an Accounts Officer to appear before the court. The last case, Mangalam Fashions Ltd & Anr vs Kolkata Municipal Corporation & Ors, is disposed of, with an order for an urgent photostat certified copy if applied for."
hyd = "This data is from three cases from the Telangana High Court. In the first case, Syed Mahmood vs T.Vijay Kumar, the Regional Joint Director of School Education rejected the petitioner's request for benefits flowing out of G.O.Ms.No.21. In the second case, Sri. B.Anil vs State Of Telangana, the petitioner leased a major portion of the 5th floor of Diamond Towers and incurred costs for improvements and infrastructure. The 5th respondent then tried to join the study centre as a partner, but was rejected by the petitioner. In the third case, D.Bala Prasada Rao, Hyderabad vs The Secretary, Energy Department, both petitioners joined the services of Agros Limited as Junior Engineers and applied for the post of Assistant Managers/District Managers, with their earlier service and benefits with Agros Limited being protected. "
openai.api_key = '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'
def assistant_speaks(output):
    global num
    num += 1
    print("Proxie : ", output)
    toSpeak = gTTS(text=output, lang='en', slow=False)
    file = str(num) + ".mp3"
    toSpeak.save(file)
    playsound.playsound(file, True)
    os.remove(file)
def get_audio():
    rObject = sr.Recognizer()
    audio = ''
    with sr.Microphone() as source:
        print("Speak...")
        # recording the audio using speech recognition
        audio = rObject.listen(source, phrase_time_limit=5)
    print("Stop.")  # limit 5 secs
    try:
        text = rObject.recognize_google(audio, language='en-US')
        print("You : ", text)
        return text
    except:
        assistant_speaks("Could not understand your audio, PLease try again !")
        return 0
def sarcasm():
    globals()['sa'] = get_audio().lower()
    response = response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Marv is a chatbot that reluctantly answers questions with sarcastic responses:\n\nYou:{str(sa)}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    stop = ["\n", " Human:", ]
    answer = response.choices[0].text.strip()
    assistant_speaks(answer)
def grammar():
    g = get_audio().lower()
    response = response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Correct this to standard English:\n\n{str(g)}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    stop = ["\n", " Human:", ]
    answer = response.choices[0].text.strip()
    assistant_speaks(answer)
def qanda():
    globals()['q'] = get_audio().lower()
    response = response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Q: {str(q)}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    stop = ["\n", " Human:", ]

    answer = response.choices[0].text.strip()
    assistant_speaks(answer)
def summarise():
    globals()['s'] = get_audio().lower()
    response = response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Summarize this for a second-grade student:\n\n{str(s)}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    stop = ["\n", " Human:", ]
    answer = response.choices[0].text.strip()

    assistant_speaks(answer)
def dall():
    globals()['dal'] = get_audio().lower()
    response = openai.Image.create(
        prompt=str(dal),
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    assistant_speaks(image_url)
def weather():
    wea = get_audio().lower()
    r = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid=5e25ee76080d529dc38f1e72624c1c60")
    json_data = r.json()
    globals()['weat'] = json_data['weather'][0]['main']
    globals()['description'] = json_data['weather'][0]['description']
    globals()['temp'] = json_data['main']['temp']
    icon = "http://openweathermap.org/img/wn/" + json_data['weather'][0]['icon'] + "@2x.png"
def news():
    url = 'https://newsapi.org/v2/everything?'
    ne = get_audio().lower()
    q = str(ne)
    pagesize = 1
    sort = 'popularity'
    key = 'a76cdc2661914ede81cadb7f8741318c'
    response = requests.get(f'https://newsapi.org/v2/everything?q={q}%20styles&pageSize=2&sortBy=popularity&apiKey={key}')
    response_json = response.json()
    article = response_json["articles"]
    author1 = []
    content1 = []
    description1 = []
    published = []
    titl = []
    url = []
    image = []
    for ar in article:
        titl.append(ar["title"])
        description1.append(ar["description"])
        url.append(ar["url"])
        author1.append(ar["author"])
        content1.append(ar["description"])
        published.append(ar["publishedAt"])
        image.append(ar["urlToImage"])
    url = str(url)[2:-2]
    globals()['title'] = str(titl)[2:-2]
    globals()['description1'] = str(description1)[2:-2]
    image = str(image)[2:-2]
def dall():
    dal = get_audio().lower()
    response = openai.Image.create(
        prompt=str(dal),
        n=1,
        size="1024x1024"
     )
    image_url = response['data'][0]['url']
    url = image_url
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.show()
if __name__ == "__main__":
    assistant_speaks("Hello, I am Proxie. The most advanced voice assistant at your service")
    while 1==1:
        text = get_audio().lower()
        if text == 0:
            continue
        if "exit" in str(text) or "bye" in str(text) or "sleep" in str(text):
            assistant_speaks("Ok bye, ")
            break
        if "grammar correction" in str(text) or "grammar" in str(text) or "grammer" in str(text):
            grammar()
        if "q and a" in str(text) or "qanda" in str(text) or "Q and A" in str(text):
            qanda()
        if "summarize" in str(text) or "summarise" in str(text) or "sumarise" in str(text) or "suma rice" in str(text) :
            summarise()
        if "Sarcasm" in str(text) or "sarcasm" in str(text) or "sir casm" in str(text):
            sarcasm()
        if "dall e" in str(text) or "dail" in str(text) or "daile" in str(text):
            dall()
        if "hyderabad high court" in str(text) or "hyderabad high qoute" in str(text):
            assistant_speaks(hyd)
        if "madras high court" in str(text):
            assistant_speaks(madras)
        if "bombay high court" in str(text):
            assistant_speaks(bom)
        if "amaravati high court" in str(text):
            assistant_speaks(andh)
        if "supreme court" in str(text):
            assistant_speaks(sup)
        if "weather" in str(text):
            weather()
            assistant_speaks(f"Weather - {weat} , Description - {description} , Temperature - {temp}")
        if "news" in str(text):
            news()
            assistant_speaks(f"{title}  {description}")
        if "dall" in str(text) or "dail" in str(text) or "image gen" in str(text) or "image generation" in str(text):
            dall()
        else:
                ur = f"http://api.brainshop.ai/get?bid=170226&key=qje9vuLTq5llXvvE&uid=[uid]&msg={str(text)}"
                r = requests.get(ur)
                json_data = r.json()
                message = json_data['cnt']
                assistant_speaks(message)
