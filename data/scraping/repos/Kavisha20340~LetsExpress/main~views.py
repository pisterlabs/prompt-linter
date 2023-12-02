from django.shortcuts import render
from django.http import HttpResponse
from .models import PatientData
from .forms import PatientForm
from django.contrib import messages
from django.db.models import Q
from methacksproject.globals import GLOBAL_FNAME, GLOBAL_LNAME
from datetime import date
from django.db.models import F
import cohere
from cohere.responses.classify import Example


#Where ALL the python functions are written

def home(request):
    global GLOBAL_FNAME
    global GLOBAL_LNAME
    GLOBAL_LNAME = None
    GLOBAL_FNAME = None
    return render(request, 'index.html', {})


def form(request):
    global GLOBAL_FNAME
    global GLOBAL_LNAME
    if request.method == "POST":
        submission = PatientForm(request.POST or None)
        if submission.is_valid():
            submission.save()
            date = request.POST['date']
            patientfName = request.POST['fname']
            patientlName = request.POST['lname']
            if GLOBAL_FNAME == None and GLOBAL_LNAME == None:
                GLOBAL_FNAME = patientfName
                GLOBAL_LNAME = patientlName
            return analyzeEntry(request, GLOBAL_FNAME, GLOBAL_LNAME, date)
        else:
            return render(request, 'form.html', {})
    else:
        return render(request, 'form.html', {})
    

def analyzeEntry(request, fname, lname, date):
    all_entries = PatientData.objects.filter(Q(fname__icontains = fname) & Q(lname__icontains = lname) & Q(date__icontains = date))
    dayEntry = ""
    for index, entry in enumerate(all_entries):
        dayEntry += str(entry)
        if len(all_entries) > 1 and index != len(all_entries)-1:
            dayEntry += ". "
    cohereClassification = responseEval(dayEntry)
    all_entries.update(mood=cohereClassification)
    search = 'Give me some suggestions to feel better for a '+ str(cohereClassification) +' mood for my scenario:' + str(dayEntry)   
    cohereGen = generateFeedback(search)
    all_entries.update(feedback=cohereGen)
    return render(request, "postForm.html", {'feedback': cohereGen})

    

def postForm(request, originalEntries, analyzedEntries):
    return render(request, "postForm.html", {'entries': originalEntries, 'analyzed': analyzedEntries})


#Function for all entries from login page
def summary(request):
    patientfName = request.POST['fname']
    patientlName = request.POST['lname']
    global GLOBAL_FNAME
    GLOBAL_FNAME = patientfName
    global GLOBAL_LNAME
    GLOBAL_LNAME = patientlName

    all_entries = PatientData.objects.filter(Q(fname__icontains = GLOBAL_FNAME) & Q(lname__icontains = GLOBAL_LNAME))
    return render(request, 'summary.html', {'entries': all_entries, 'first': GLOBAL_FNAME, 'last': GLOBAL_LNAME})


def viewEntries(request):
    if GLOBAL_LNAME == None and GLOBAL_FNAME == None:
        return HttpResponse("No Entries")
    else:
        all_entries = PatientData.objects.filter(Q(fname__icontains = GLOBAL_FNAME) & Q(lname__icontains = GLOBAL_LNAME))
        return render(request, 'summary.html', {'entries': all_entries, 'first': GLOBAL_FNAME, 'last': GLOBAL_LNAME})


def filterNew(request):
    all_entries = PatientData.objects.filter(Q(fname__icontains = GLOBAL_FNAME) & Q(lname__icontains = GLOBAL_LNAME)).order_by('-date') #asciending order
    return render(request, 'summary.html', {'entries': all_entries, 'first': GLOBAL_FNAME, 'last': GLOBAL_LNAME})


def filterOld(request):
    all_entries = PatientData.objects.filter(Q(fname__icontains = GLOBAL_FNAME) & Q(lname__icontains = GLOBAL_LNAME)).order_by('date') #asciending order
    return render(request, 'summary.html', {'entries': all_entries, 'first': GLOBAL_FNAME, 'last': GLOBAL_LNAME})

#classification 
co = cohere.Client('eSqjhY9UzoLkThXayCZaW4QLKbr7YFnQgIseI1rX') # This is your trial API key # This is your trial API key
def responseEval(msg):
  response = co.classify(
  model='large',
  inputs=[msg],
  examples=[Example("Today, I woke up feeling empty and hopeless. I don\'t know what\'s wrong with me, but I can\'t shake this feeling of sadness. I tried to distract myself by doing things I enjoy, but nothing seems to help.", "Melancholy"), 
            Example("I got some bad news today, and I\'m devastated. It feels like everything is falling apart around me. I don\'t know how to process this information, and I don\'t know who to turn to for support. I just want to curl up in a ball and cry.", "Melancholy"), 
            Example("I\'m so tired of feeling this way. It\'s like a dark cloud has been following me around for weeks, and I can\'t seem to shake it. I don\'t know how to lift myself out of this funk, and I\'m afraid I\'ll be stuck here forever.", "Melancholy"), 
            Example("I had a panic attack today, and it was terrifying. I felt like I was going to die, and I couldn\'t catch my breath. Even after it was over, I couldn\'t calm down. I\'m afraid it will happen again, and I don\'t know how to prevent it.", "Anxiety"), 
            Example("Today, my anxiety was at an all-time high. I woke up with a knot in my stomach, and I couldn\'t shake the feeling of impending doom. Every little thing felt like a threat, and I couldn\'t stop worrying about everything.", "Anxiety"), 
            Example("I\'m so anxious about the future. I keep thinking about all the things that could go wrong, and I can\'t seem to stop. It\'s like my mind is stuck in a loop, and I can\'t break free.", "Anxiety"), 
            Example("I\'m so angry right now, I could scream. I feel like everything is going wrong, and there\'s nothing I can do to fix it. I\'m tired of feeling helpless and frustrated all the time.", "Anger"), 
            Example("I\'m angry at myself for letting things get this bad. I should have known better, but I ignored all the warning signs. Now I\'m paying the price, and it feels like there\'s no way out.", "Anger"), 
            Example("The world feels so unfair right now, and it\'s making me angry. There are so many injustices happening every day, and it feels like no one is doing anything about it. I don\'t know how to channel this anger into something productive.", "Anger"), 
            Example("I\'m so confused about what I want to do with my life. I thought I had everything figured out, but now I\'m not so sure. I don\'t know if I should keep pursuing my current path or if I should try something new.", "Confusion"), 
            Example("I\'m confused about my relationships. I don\'t know if the people in my life are good for me or if I should distance myself. I\'m afraid of hurting them, but I\'m also afraid of staying in toxic situations.", "Confusion"), 
            Example("Today, I\'m grateful for my health. I woke up feeling energized and ready to take on the day, and I know that\'s not something everyone gets to experience.", "Gratitude"), 
            Example("I\'m thankful for my supportive friends and family. They have been there for me through thick and thin, and I know I wouldn\'t be where I am today without them.", "Gratitude"), 
            Example("I\'m grateful for my job. It\'s not always easy, but it provides me with stability and the opportunity to grow and learn new things.", "Gratitude"), 
            Example("Today was an amazing day! I woke up feeling refreshed and energized, ready to tackle the challenges of the day. I had a productive morning and was able to complete all of the tasks on my to-do list. What made the day even better was that I received an email from my dream job saying that they were interested in interviewing me. I feel hopeful and excited about what the future holds!", "Hopefulness"), 
            Example("Today, I went for a walk in the park and it was such a beautiful day. The sun was shining and the flowers were in full bloom. I felt a sense of peace and calm that I haven\'t felt in a long time. As I walked, I thought about all of the possibilities that the future holds and felt hopeful that everything will work out in the end. I am grateful for moments like this that remind me to appreciate the present and look forward to the future.", "Hopefulness"), 
            Example("Today was a day of new beginnings. I started a new job that I am really excited about and feel hopeful about the opportunities it will bring. I met some great people and feel like I am already starting to fit in. After work, I went for a run and felt a sense of accomplishment and pride. I am starting to see the results of all the hard work I have been putting in and it feels great. Today was a reminder that with hard work and dedication, anything is possible.", "Hopefulness"), 
            Example("I ran into an old friend today and found out that they had just bought a new house. As they showed me pictures of the beautiful home with a huge backyard and spacious rooms, I couldn\'t help but feel a tinge of envy. I\'ve been struggling to make ends meet and the thought of owning a home seems like an impossible dream. Seeing someone else achieve what I\'ve been striving for only made me feel worse.", "Envy"), 
            Example("Today, I saw a coworker get promoted to a position that I\'ve been working hard for. While I am happy for them and know that they deserve the promotion, I couldn\'t help but feel a sense of envy. I\'ve been working just as hard and feel like I\'ve been overlooked. I know I shouldn\'t compare myself to others, but it\'s hard not to feel envious when someone else gets something you want.", "Envy"), 
            Example(" I saw pictures on social media of an old classmate who is now traveling the world, visiting exotic places and experiencing new cultures. As I scrolled through the pictures, I felt envious of their freedom and adventure. I\'ve been stuck in the same routine and feel like I\'m missing out on all the excitement. Seeing someone else living the life I want only made me feel more envious and resentful.", "Envy"), 
            Example("Today was a simple, yet wonderful day. I spent the morning reading a good book and drinking a cup of coffee. In the afternoon, I went for a walk in the park and enjoyed the fresh air and sunshine. As I reflect on the day, I feel content and grateful for the little things in life that bring me joy. It\'s moments like these that make me appreciate the present and feel content with where I am in life.", "Content"), 
            Example("Today, I spent the day with my family, just enjoying each other\'s company. We had a barbecue, played games, and talked about our lives. As I watched my kids running around and laughing, I felt a deep sense of contentment. It\'s times like these that I realize how lucky I am to have a loving family and how much they mean to me.", "Content"), 
            Example("Today, I accomplished a goal that I\'ve been working towards for a long time. I finally finished writing a book that I\'ve been working on for years. As I hit the \"send\" button to submit the manuscript, I felt a sense of contentment and pride. It\'s a great feeling to achieve something that I\'ve worked so hard for and to see the fruits of my labor.", "Content"), 
            Example("Today was one of those days where everything seemed to go wrong. I woke up late, spilled coffee on my shirt, and missed an important meeting at work. The rest of the day was spent playing catch-up and trying to fix my mistakes. As I sit here, exhausted and stressed, I can\'t help but feel like everything is falling apart.", "Stressed"), 
            Example("Today, I received some unexpected bills in the mail that I wasn\'t prepared for. I\'ve been struggling financially and the added stress of these bills is overwhelming. I feel like I\'m drowning in debt and can\'t seem to catch a break. The weight of these financial burdens is taking a toll on my mental and physical health, and I\'m not sure how to cope.", "Stressed"), 
            Example("The holidays are supposed to be a time of joy and celebration, but for me, it\'s just another source of stress. I have a long list of gifts to buy, parties to attend, and family obligations to fulfill. The thought of all of these commitments is overwhelming and I feel like I can\'t keep up. Instead of feeling excited for the holidays, I\'m filled with stress and anxiety.", "Stressed"),
            Example("The holidays are supposed to be a time of joy and celebration, but for me, it\'s just another source of stress. I have a long list of gifts to buy, parties to attend, and family obligations to fulfill. The thought of all of these commitments is overwhelming and I feel like I can\'t keep up. Instead of feeling excited for the holidays, I\'m filled with stress and anxiety.", "Stressed"),
            Example("Today has been a rough day. I feel completely overwhelmed and stressed out. It seems like everything is piling up on me all at once. Work has been especially demanding lately, and I'm having trouble keeping up with everything. I'm also dealing with some personal issues that have been weighing on my mind. I can't seem to shake this feeling of anxiety, and it's making it hard for me to focus on anything. My heart is racing, and I feel like I'm on edge all the time. I know I need to find a way to manage this stress before it completely takes over my life.", "Stressed"),
            Example("I feel like I'm constantly running behind schedule, and it's stressing me out. No matter how hard I try to manage my time, there just never seems to be enough of it. I'm juggling so many different tasks and responsibilities, and it's starting to take a toll on me. I feel like I'm always on the go, and I never have a chance to catch my breath. I'm exhausted all the time, and I'm struggling to keep up with everything. I know I need to figure out a way to manage my time more effectively if I want to get control of this stress.", "Stressed"),
            Example("I'm feeling completely burned out today. I've been working so hard lately, and it seems like no matter how much effort I put in, I just can't get ahead. I'm feeling stressed out all the time, and I'm starting to lose my motivation. I don't feel like I have anything left to give, and it's making me feel really down. I know I need to take a step back and recharge my batteries before this stress completely consumes me. I need to find a way to manage my workload and take care of myself at the same time.", "Stressed"),
            Example("Today was another gloomy day. The weather outside matched my mood perfectly. I couldn\'t shake off this feeling of sadness that has been haunting me for days. It\'s like a heavy weight on my chest that won\'t go away. I tried to distract myself by reading, watching movies, and listening to music, but nothing seems to work. Maybe tomorrow will be a better day.", "Melancholy"),
            Example("I woke up feeling empty and hopeless today. It\'s hard to explain why I feel this way. It\'s like there\'s a void inside me that nothing can fill. I don\'t have the motivation to do anything, and even the simplest tasks seem overwhelming. I miss the days when I used to feel happy and alive. I wonder if I will ever feel that way again.", "Melancholy"),
            Example("The world seems so dark and cruel lately. Everywhere I look, I see pain and suffering. It\'s hard to believe that there\'s any goodness left in the world. Sometimes, I feel like giving up and surrendering to the darkness. But I know I can\'t do that. I have to keep going, even if it feels like I\'m fighting a losing battle. Maybe someday, the sun will shine again.", "Melancholy"),
            Example("I don\'t know what to do. I\'m so confused about my life and my future. I thought I had everything figured out, but now I\'m not so sure. I don\'t know if I\'m on the right path, or if I should make a drastic change. I feel lost and alone, with no one to turn to for guidance. I wish I had a clear direction, but all I see is a maze of uncertainty.", "Confusion"),
            Example("Today was a strange day. Everything seemed out of place, and I couldn\'t make sense of it. People were saying one thing and doing another, and I didn\'t know who to believe. I felt like I was in a dream, where nothing is real and everything is confusing. I wish I could just wake up and have everything be clear again.", "Confusion"),
            Example("I\'m in a state of constant confusion lately. It\'s like my brain is in a fog, and I can\'t see anything clearly. I don\'t know what I want, what I need, or what I should do. It\'s all a jumbled mess in my head, and I don\'t know how to sort it out. I wish I had a map or a guidebook to help me navigate through this confusion, but all I have is my own intuition, which seems to be failing me.", "Confusion"),
            Example("Today, I am grateful for my health. I am so lucky to have a body that can move, breathe, and function properly. I am grateful for the ability to wake up each day and start anew. I will try to take care of myself by nourishing my body and being mindful of my health.", "Gratitude"),
            Example("I am grateful for my friends and family. They bring so much joy, laughter, and love into my life. I appreciate their support and encouragement, even when I am feeling down. I am blessed to have such wonderful people in my life.", "Gratitude"),
            Example("Today, I am grateful for nature. I am thankful for the beauty of the trees, the flowers, and the sky. I am grateful for the fresh air and the sunlight that nourishes my body and soul. I will try to spend more time outdoors, connecting with nature and appreciating its wonders.", "Gratitude"),
            Example("Today, I am grateful for my education. I am blessed to have access to knowledge and opportunities that many people in the world do not have. I am grateful for the chance to learn, to grow, and to become a better version of myself. I will strive to use my education for good, to make a positive impact on the world around me.", "Gratitude"),
            Example("Today, I am feeling content. I am grateful for the simple pleasures in life, like spending time with loved ones, enjoying a good meal, and watching the sunset. I feel at peace with myself and the world around me. I will try to hold onto this feeling of contentment and spread positivity wherever I go.", "Content"),
            Example("I am content with where I am in life. I am proud of my accomplishments, and I am excited for what the future holds. I have set goals for myself, and I am working hard to achieve them. I am grateful for the opportunities that have come my way and the people who have supported me along the way.", "Content"),
            Example("Today, I am super content with myself. I am learning to love and accept myself for who I am, flaws and all. I am grateful for my strengths and my weaknesses, as they make me who I am. I will try to practice self-care and self-compassion, and to treat myself with kindness and respect.", "Content")])
  return response.classifications[0].prediction

#generate
co = cohere.Client('eSqjhY9UzoLkThXayCZaW4QLKbr7YFnQgIseI1rX') # This is your trial API key # This is your trial API key
def generateFeedback(msg):
    response = co.generate(
    model='command-xlarge-nightly',
    prompt=msg,
    max_tokens=300,
    temperature=0.9,
    k=0,
    stop_sequences=[],
    return_likelihoods='NONE')
    print('Prediction: {}'.format(response.generations[0].text))
    return response.generations[0].text


def analyzeAll(request):
    allPatientEntriesToday = PatientData.objects.filter(Q(date__icontains = date.today()))

    totalEntries = len(allPatientEntriesToday)
    moodDict = {"Melancholy": 0, "Anxiety": 0, "Anger": 0, "Confusion": 0, "Gratitude": 0, "Hopefulness": 0, "Envy": 0, "Content": 0, "Stressed": 0}
    for key in moodDict:
        moodEntries = PatientData.objects.filter(Q(mood__icontains = key) & Q(date__icontains = date.today()))
        totalmoodEntries = len(moodEntries)
        if totalmoodEntries == 0 or totalEntries == 0:
            moodDict[key] = 0
        else:
            percentage = int((totalmoodEntries/totalEntries) * 100)
            moodDict[key] = percentage

    allFeedback = ""
    for index, entry in enumerate(allPatientEntriesToday):
        allFeedback += entry.feedback
        if index != totalEntries-1:
            allFeedback += ". " 

    search = 'Given all the mental health and self care tips: ' + allFeedback + ', give me ONE mental health tip that summarizes everything.'
    cohereGen = generateFeedback(search)

    return render(request, 'community.html', {"allDayEntries": moodDict, "finalFeedback": cohereGen})



