import re
import random
import spacy
import DataBase
import cohere

nlp = spacy.load("en_core_web_lg")
entity_list = []

gender = ['male', 'female', 'transgender ', 'trans', 'cisgender', 'cis', 'nonbinary', 'non binary', 'genderqueer',
          'gender queer', 'queer', 'agender', 'genderfluid', 'gender fluid', 'bigender', 'bi gender', 'twospirit',
          'two spirit', 'androgynous', 'neutrois', 'demigender', 'demi gender', 'genderquestioning',
          'gender questioning', 'gendernonconfirming', 'gender nonconforming', 'pangender', 'pan gender', 'thirdgender',
          'third gender', 'genderflux', 'gender flux', 'intergender', 'inter gender', 'multigender', 'multi gender',
          'polygender', 'poly gender', 'gender variant', 'two souled', 'gender expansive', 'gendervague',
          'gender vague', 'femme', 'butch', 'genderfluid femme', 'gender fluid femme', 'genderfluid butch',
          'gender fluid butch', 'demiboy', 'demi boy', 'demigirl', 'demi girl', 'agenderflux', 'agender flux',
          'genderqueer femme', 'gender queer femme', 'genderqueer butch', 'gender queer butch', 'gender neutral',
          'bi genderqueer', 'bi gender queer', 'trigender', 'tri gender', 'graygender', 'gray gender',
          'trans masculine', 'transmasculine', 'trans feminine', 'transfeminine', 'androgyne', 'fluxgender',
          'flux gender', 'maverique', 'null gender', 'vapogender', 'vapo gender', 'libragender', 'libra gender',
          'aporagender', 'apora gender', 'ambonec', 'genderfae', 'faegender', 'fae gender', 'epigender', 'epi gender',
          'genderfluid demigirl', 'gender fluid demigirl', 'genderfluid demi girl', 'genderfluid demiboy',
          'gender fluid demiboy', 'genderfluid demi boy', 'juxera', 'novigender', 'proxvir', 'quoigender',
          'venusgender', 'xenogender', 'zerogender', 'demiflux', 'echogender', 'quoi gender', 'venus gender',
          'xeno gender', 'zero gender', 'demi flux', 'echo gender', 'gender questioning', 'gender nonbinary',
          'abimegender', 'astralgender', 'autigender', 'caelgender', 'deliciagender', 'demifluid', 'enbyfluid',
          'fictigender', 'glimragender', 'librafeminine', 'lunagender', 'abime gender', 'astral gender', 'auti gender',
          'cael gender', 'delicia gender', 'demi fluid', 'enby fluid', 'ficti gender', 'glimra gender',
          'libra feminine', 'luna gender', 'masculine of center', 'mascfluid', 'mirrorgender', 'paragender',
          'stellargender', 'masc fluid', 'mirror gender', 'para gender', 'stellar gender']
all_events = ['2014 Gay Games', '4th World Outgames 2017', 'Ascension Fire Island', 'Atlanta Pride', 'Atlanta Pride ',
              'Atlantis Allure all gay cruise ', 'Atlantis Americas gay cruise', 'Atlantis Caribbean Cruise',
              'Atlantis Caribbean all gay cruise', 'Atlantis Celebrity Edge gay cruise ',
              'Atlantis LA Mexico gay cruise', 'Atlantis Mexican Riveira Cruise ', 'Atlantis Mexico Halloween cruise',
              'Atlantis Mexico cruise', 'Atlantis Oasis all gay cruise', 'Atlantis all gay cruise New York',
              'Atlantis gay cruise on Harmony', 'Atlantis gay cruise on NCL Prima', 'Austin Gay Pride',
              'Barbra Back To Brooklyn', 'Bear Bash', 'Bear Week', 'Black Pride at the Beach', 'Black XXXMas ',
              'Blatino Oasis', 'Boston Pride', 'Burning man', 'Capital pride', 'Carnival', 'Cherry', 'Chicago Pride',
              'Claw Leather Awareness', 'Cleveland Pride', 'Cruise4Bears NYC to Caribbean Cruise', 'DC Black Pride',
              'Dallas Gay Pride', 'Dallas Pride Weekend', 'Dance at the Pier', 'Disney Gay Days',
              'Elevation Mammoth gay ski week', 'Evolve Vegas NYE', 'Folsom Street East', 'Folsom Street Fair',
              'Freedom Caribbean Cruise', 'Gay Days Anaheim', 'Gay Days Las Vegas', 'Gay Days One Mighty Weekend',
              'Gay Days Reunion Weekend', 'Gay Easter Parade', 'Gay Halloween Streetparty', 'Gay Mardi Gras',
              'Gay Mardi Gras ', 'Gay Pride Fest Denver', 'Gay Pride Houston', 'Gay Pride New Orleans',
              'Gay Pride at Burning man ', 'Gay and Lesbian Ski Week', 'Greater Palm Springs Pride', 'Halloween',
              'Halloween ', 'Harlem Pride', 'Harlem Pride 2011', 'Honolulu gay pride', 'Hustlaball Las Vegas',
              'Independence Weekend', 'International Bear Convergence IBC', 'International Mr Leather',
              'Key West Pride', 'Key West PrideFest', 'Key West PrideFest ', 'Kiki Cruise', 'LA Fetish Pride',
              'Las Vegas Pride', 'Long Beach Pride', 'Los Angeles Pride', 'Matinee Festival',
              'Matinee Las Vegas Festival', 'Memorial Weekend', 'Miami Beach gay pride', 'Mid Atlantic Leather Weekend',
              'Mid Atlantic Leather Weekend ', 'Motor City Pride', 'NY AIDS Walk', 'NYC Black Pride: Explosion',
              'NYC Gay Pride', 'NYC Gay Pride / Worldpride', 'Northalsted Market Days', 'Oakland Gay Pride',
              'Olivia Lesbian Cruise', 'One Magical Weekend', 'One Mighty Weekend', 'Orlando Pride', 'Outfest',
              'Paradise Circuit Festival', 'Philly Gay Pride', 'Phoenix LGBT+ gay pride festival', 'Phoenix gay pride',
              'Pig Week', 'Pig Week ', 'Portland Pride', 'Pride @ the beach', 'Pride @ the beach ',
              'Pride Fort Lauderdale', 'Pride Fort Lauderdale ', 'Pride by the beach', 'Pride of the Americas',
              'Pridefest South Florida', 'Purple Party', 'RSVP Caribbean Cruise', 'RSVP Caribbean Cruise ',
              'RSVP Caribbean gay cruise', 'RSVP Hawaiian Islands Cruise', 'RSVP Mardi Gras Caribbean Cruise',
              'RSVP gay Alaska cruise', 'Razzle Dazzle', 'Rhode Island Pride', 'Ripped', 'San Diego Pride',
              'San Francisco Pride', 'Sand Blast Weekend', 'Seattle Pride', 'Sizzle', 'Southern Decadence',
              'Splash Days', 'Splash Days ', 'Summer Camp', 'The Black Party', 'Tropical Heat', 'Twin Cities Pride',
              'Up Your Alley', 'Vacaya PTown gay cruise', 'White Party Palm Springs', 'White Party Week', 'Winter Heat',
              'Winter Party', 'Winter Party Miami', 'Winter Party Miami ', 'Wonder World']
all_communities = ['1736 Family Crisis Center', '5 Keys Charter School at the LA LGBT Center', 'ACLU - Socal',
                   'ADORE-LA', 'AHF Healthcare Center - Downtown LA', 'AHF Healthcare Center - Hollywood',
                   'API Equality', 'APLA Health at Tarzana Treatment Center',
                   'Against The Stream Buddhist Meditation Society', 'Alexandria House', 'American Catholic Church',
                   'Ascencia', 'Asian Pacific AIDS Intervention Team (APIAT)',
                   'Being Alive - People with HIV/AIDS Action Coalition', 'Beth Chayim Chadashim', 'Bienestar',
                   'Bienestar SABORES Youth Program - Pomona',
                   'Bienestar SABORES Youth Program-East Los Angeles Center',
                   'Bienestar SABORES Youth Program-Hollywood Center',
                   'Bienestar SABORES Youth Program-Long Beach Center',
                   'Bienestar SABORES Youth Program-South Los Angeles Center', 'Black AIDS Institute',
                   'California STD/HIV Prevention Training Center', 'Catholic Ministry with Lesbian and Gay Persons',
                   'Children of the Night', 'Children’s Hospital Los Angeles-Tranny Rockstar',
                   'Christ Chapel of the Valley', 'Colors LGBTQ Youth Counseling Center',
                   'Common Ground-West Side HIV Community Center', 'Congregation Kol Ami', 'Covenant House California',
                   'Destinations for Teens', 'Didi Hirsch Mental Health Services (18 & Over)', 'Dignity/LA',
                   'Emergency Overnight Bed Program - Youth Center On Highland', 'Everyone Is Gay',
                   'Friends Community Center', 'Friends of Project 10', 'GSA Network', 'Gleicher/Chen Health Center',
                   'Global Truth Center', 'HOPE CENTER- Transitional Age Youth (TAY) ages 16-25',
                   'Hollywood Lutheran Church', 'Hollywood United Methodist Church', 'IKAR',
                   'ILP (Independent Living Program)', 'Inspire Spiritual Community', 'JQ International',
                   'Jordan/Rustin Coalition', 'LA Gender Center', 'LA LGBT Center Legal Services',
                   'LA LGBT Health Center Services', 'LGBT Center OC', 'Lambda Legal', 'Latino Equality Alliance',
                   'LifeWorks - The Village at Ed Gould Plaza', 'LifeWorks Scholarships', 'Long Beach- Safe Refuge',
                   'Los Angeles County Department of Public Health - Torrance Health Center',
                   'Los Angeles House of Ruth', 'Los Angeles LGBT Center - McDonald/Wright Building',
                   'Los Angeles LGBT Center - The Village at Ed Gould Plaza',
                   'Los Angeles LGBT Center - Youth Center on Highland', 'Metropolitan Community Church',
                   'Metropolitan Community Church/United Church of Christ in the Valley', 'Mi Centro',
                   'Minority AIDS Project', 'Muslims for Progressive Values', 'My Friend’s Place (MFP)',
                   'My Transwellness Center', 'Neighborhood Unitarian Universalist Church - Pasadena',
                   'New Abbey Church', 'ONE National Gay & Lesbian Archives at the USC Libraries',
                   'Out of the Closet Thrift Store', 'PFLAG Long Beach', 'PFLAG Los Angeles', 'PFLAG Pasadena',
                   'PFLAG SGV Asia Pacific Islander', 'PFLAG Santa Clarita', 'Pedro Zamora Youth HIV Clinic',
                   'Penny Lane Centers - Commerce', 'Penny Lane Centers - Headquarters',
                   'Penny Lane Centers - North Hollywood', 'Penny Lane Centers - Palmdale',
                   'Planned Parenthood - Basics Baldwin Hills/Crenshaw Health Center',
                   'Planned Parenthood - Dorothy Hecht Health Center', 'Planned Parenthood - Eagle Rock Health Center',
                   'Planned Parenthood - East Los Angeles Health Center',
                   'Planned Parenthood - Hollywood Health Center',
                   'Planned Parenthood - S. Mark Taper Foundation Center for Medical Training',
                   'Planned Parenthood - Stoller Filer Health Center',
                   'Planned Parenthood - West Hollywood Health Center',
                   'Planned Parenthood Los Angeles - South Bay Health Center', 'Point Foundation', 'Project Angel Food',
                   'Reach LA', 'Renovados Por El Poder De Dios', 'San Fernando Valley Rescue Mission',
                   'San Gabriel Valley LGBTQ Center', 'South Bay LGBT Center', 'St Thomas the Apostle Episcopal Church',
                   "St. John's Well Child and Wellness Center", 'StepUp', 'The Center Long Beach',
                   'The Center for Transgender Youth Health and Development', 'The David Geffen Center',
                   'The LGBT Community Center of the Desert', 'The Lavender Effect',
                   'The Los Angeles LGBT Center - WeHo', 'The OUTreach Center',
                   'The San Diego LGBT Community Center - Centre Street - Hillcrest',
                   'The San Diego LGBT Community Center - Hillcrest Youth Center',
                   'The San Diego LGBT Community Center - Sunburst Youth Housing Project', 'UCLA EMPWR Program',
                   'Unitarian Universalist Church of Studio City', 'Valley Teen Clinic',
                   'Village Family Services - San Fernando Valley', 'Walden Family Services',
                   'West Hollywood United Church of Christ', 'Youth Center on Highland',
                   'Youth Emerging Stronger (Formerly LAYN)', 'Youth Emerging Stronger (formerly LAYN)']

bot_template = "Unite_Bot : {0}"
user_template = "User : {0}"
patterns = [
    [[
        "hi",
        "hello",
        "hey",
    ], [
        "Hi there!",
        "Hello, how can I assist you today?",
        "Hey, what's on your mind?",
    ]],
    [["what's up",
      "how are you"],
     ["Not much, how about you?",
      "I'm doing well, thanks for asking!"]],

    [[
        "bye",
        "goodbye",
        "see you later",
        "talk to you soon"
    ], [
        "Goodbye!",
        "Take care!",
        "See you later!",
        "Have a great day!"
    ]],
    [[
        "how do i come out to my parents?",
        "what should i do if i'm afraid to come out?",
        "i'm scared to tell my friends i'm lgbtq",
        "how do i know if it's safe to come out?",
        "what if my family doesn't accept me?"
    ], [
        "Coming out can be scary, but it's important to be true to yourself. If you don't feel safe or comfortable coming out, it's okay to wait until you are ready. You could also reach out to a trusted friend or LGBTQ support group for guidance.",
        "It's important to remember that you are not alone. Many people struggle with coming out, and it can take time to find the right moment and the right words. You might find it helpful to practice what you want to say, or to write a letter if you're having trouble speaking in person.",
        "It's okay to take your time with coming out, and to only share your identity with those who you feel safe and comfortable around. You could also consider seeking support from an LGBTQ center or therapist.",
        "It's important to prioritize your safety when coming out. If you're not sure if it's safe to come out, you might want to talk to a trusted adult or LGBTQ support group for guidance. You could also consider coming out in a public place or with a trusted friend or family member present.",
        "If your family doesn't accept you, it can be painful and difficult. It's important to remember that you are still valid and deserving of love and respect. You might want to reach out to an LGBTQ support group or a therapist for guidance and support."
    ]],
    [[
        "where can i find lgbtq support?",
        "i need help with lgbtq issues",
        "how can i find an lgbtq therapist?",
        "are there any lgbtq events happening?",
        "can you recommend an lgbtq-friendly doctor?"
    ], [
        "There are many resources available for LGBTQ support, including local community centers, online forums, and LGBTQ-focused therapy. You can also check out LGBTQ events in your area, or ask for recommendations from friends or your healthcare provider.",
        "It's important to prioritize your mental health and well-being, especially if you're struggling with LGBTQ-related issues. You might find it helpful to seek out a therapist who is experienced in working with LGBTQ clients. There are also online therapy options available if you prefer to talk to someone from the comfort of your own home.",
        "There are many resources available to you, depending on your specific needs.",
        "You can try searching online for local LGBTQ centers or support groups, or you can ask your healthcare provider for recommendations on LGBTQ-friendly therapists or doctors.",
        "You might also want to check out LGBTQ events in your area, as these can be great opportunities to connect with others in the community.",
        "Please let me know if there's anything specific I can help you with."
    ]],
    [[
        "how do i come out to my family?",
        "what if my family doesn't accept me as lgbtq?",
        "should i come out to my family?",
        "how can i tell my family i'm lgbtq?"
    ], [
        "Coming out to family can be difficult, but it's important to be true to yourself. It's best to choose a time and place where you feel safe and comfortable, and try to express your feelings and identity honestly. Remember that their reaction is not a reflection of your worth, and it may take time for them to adjust and understand. It can be helpful to seek support from a therapist or an LGBTQ support group."
    ]],
    [[
        "what is gender identity?",
        "how do i know my gender identity?",
        "what if i don't identify with the gender i was assigned at birth?",
        "what is gender dysphoria?"
    ], [
        "Gender identity is a person's internal sense of being male, female, or something else. It's different from biological sex, which is assigned at birth based on physical characteristics. Gender identity can be fluid and can vary from person to person.",
        "Discovering your gender identity can be a journey, and it's important to give yourself time and space to explore your feelings. Some people know their gender identity from a young age, while others may not fully understand it until later in life. It's okay to question and explore your gender identity at any age.",
        "If you don't identify with the gender you were assigned at birth, it's important to remember that you are not alone. Many people experience gender dysphoria, which is a distressing feeling that occurs when there is a mismatch between a person's gender identity and their assigned sex. It's important to seek support from an LGBTQ-friendly therapist or support group if you're struggling with gender dysphoria.",
        "Gender dysphoria is a medical diagnosis used to describe the distress that can occur when a person's gender identity doesn't align with the gender they were assigned at birth. It's not a mental illness, and it's treatable through gender-affirming therapies and medical interventions. It's important to seek support from a qualified healthcare provider or therapist if you're experiencing gender dysphoria."
    ]],
    [[
        "what is sexual orientation?",
        "how do i know my sexual orientation?",
        "what if i'm attracted to people of the same gender?",
        "what if i'm not sure about my sexual orientation?"
    ], [
        "Sexual orientation refers to a person's pattern of emotional, romantic, and/or sexual attractions to men, women, both genders, or neither gender. It's a normal and natural variation of human sexuality.",
        "Discovering your sexual orientation can be a process, and it's important to give yourself time and space to explore your feelings. Some people know their sexual orientation from a young age, while others may not fully understand it until later in life. It's okay to question and explore your sexual orientation at any age.",
        "If you're attracted to people of the same gender, it's important to remember that you are not alone. Many people identify as lesbian, gay, bisexual, or queer. It's important to seek support from an LGBTQ-friendly therapist or support group if you're struggling with your sexual orientation.",
        "It's okay to not be sure"
    ]],
    [[
        "what are preferred pronouns?",
        "what do pronouns mean?",
        "why are pronouns important?",
        "how can i ask someone's pronouns?"
    ], [
        "Preferred pronouns are the pronouns that someone chooses to use to refer to themselves. Some examples of pronouns are he/him, she/her, they/them, ze/hir. It's important to respect people's chosen pronouns as a way of honoring their gender identity.",
        "Pronouns are the words we use to refer to someone without using their name. They can be important because they can signal someone's gender identity, and using the wrong pronouns can be hurtful and disrespectful.",
        "Pronouns are important because they help to affirm someone's gender identity and show respect for their identity. Using the correct pronouns is a way of honoring someone's identity and promoting inclusivity.",
        "Asking someone's pronouns can be as simple as saying, 'What pronouns do you use?' or 'Can you remind me of your pronouns?' It's important to ask in a respectful and non-judgmental way, and to avoid assuming someone's pronouns based on their appearance."
    ]],
    [[
        "what does it mean to be transgender?",
        "how do i support a transgender friend or family member?",
        "what are some challenges that transgender people face?",
        "what is gender reassignment surgery?"
    ], [
        "Being transgender means that a person's gender identity does not align with the gender they were assigned at birth. It's important to support transgender individuals by using their preferred name and pronouns, and by respecting their gender identity. It can also be helpful to educate oneself on transgender issues and to advocate for transgender rights.",
        "Supporting a transgender friend or family member can involve actively listening to them, using their preferred name and pronouns, and being an ally in their journey. It can also be helpful to educate oneself on transgender issues and to advocate for transgender rights.",
        "Transgender individuals can face discrimination, harassment, and violence. It's important to be aware of these challenges and to advocate for transgender rights. Transgender individuals may also face barriers to accessing healthcare, employment, and housing.",
        "Gender reassignment surgery, also known as gender confirmation surgery, is a surgical procedure that can help transgender individuals affirm their gender identity. Not all transgender individuals choose to have surgery, and it's important to respect their individual choices and experiences."
    ]],
    [[
        "how do periods affect transgender men?",
        "what are the best menstrual products for non-binary people?",
        "can hormone therapy affect menstrual cycles?"
    ], [
        "Transgender men may experience menstrual cycles even after starting hormone therapy. It's important to use the menstrual products that work best for you and to talk to your doctor about any changes in your menstrual cycle due to hormone therapy."
    ]],
    [[
        "how can i prevent sexually transmitted infections as a gay man?",
        "what are the symptoms of syphilis in women?",
        "how often should i get tested for stis?"
    ], [
        "Using condoms or other barriers during sexual activity can help prevent the spread of sexually transmitted infections. It's also important to get regular STI testing and to talk to your healthcare provider about any concerns you may have.",
        "The symptoms of syphilis can vary depending on the stage of the infection, but may include sores or rash, fever, and fatigue. It's important to get tested for syphilis and other STIs regularly if you're sexually active."
    ]],
    [[
        "what are some good hygiene practices for trans people?",
        "how can i safely bind my chest?",
        "what are the best ways to clean sex toys?"
    ], [
        "Good hygiene practices for trans people may include using gentle, non-irritating products, avoiding douching or other harsh cleaning methods, and taking care when binding or tucking to avoid skin irritation or injury.",
        "It's important to use a properly fitting binder and to take breaks from binding to avoid chest pain or breathing difficulties. You can also try alternative methods of chest binding, such as using compression shirts or sports bras.",
        "To clean sex toys, use warm water and a gentle, non-abrasive soap. Make sure to follow the manufacturer's instructions for cleaning and storing your toys, and avoid sharing them with partners to prevent the spread of STIs."
    ]],
    [[
        "how can i find a therapist who specializes in lgbtq issues?",
        "what are some ways to cope with dysphoria?",
        "how can i support a friend who is struggling with their mental health?"
    ], [
        "You can search for LGBTQ-friendly therapists online or ask for recommendations from LGBTQ community organizations or healthcare providers. It's important to find a therapist who is supportive and knowledgeable about the unique experiences of LGBTQ individuals.",
        "Some ways to cope with dysphoria may include using affirming language and clothing, connecting with other trans or gender-nonconforming individuals, and exploring options for hormone therapy or surgery.",
        "If you have a friend who is struggling with their mental health, it's important to offer support and encouragement while also respecting their boundaries and autonomy. You can encourage them to seek professional help and provide resources or information if they are interested."
    ]],
    [[
        "what are the common stis in the lgbtq community?",
        "how can i prevent getting stis?",
        "what should i do if i suspect i have an sti?"
    ], [
        "Common STIs in the LGBTQ community include HIV, gonorrhea, chlamydia, syphilis, and herpes. You can prevent getting STIs by using condoms or dental dams during sex, getting regular STI screenings, and limiting the number of sexual partners you have. If you suspect you have an STI, you should see a healthcare provider for testing and treatment."
    ]],
    [[
        "what is prep?",
        "how effective is prep?",
        "how can i get access to prep?"
    ], [
        "PrEP (pre-exposure prophylaxis) is a medication that can help prevent HIV transmission. When taken as prescribed, PrEP is highly effective at reducing the risk of getting HIV. You can talk to your healthcare provider or local health clinic to see if PrEP is right for you and how to access it."
    ]],
    [[
        "what are some mental health resources available for lgbtq individuals?",
        "how can i find an lgbtq-friendly therapist?",
        "what should i do if i'm struggling with my mental health?"
    ], [
        "There are many mental health resources available for LGBTQ individuals, including therapy, support groups, and hotlines. You can find an LGBTQ-friendly therapist by searching online directories, asking for recommendations from friends or healthcare providers, or contacting LGBTQ organizations. If you're struggling with your mental health, it's important to reach out for help and support."
    ]],
    [[
        "what are some options for body hair removal?",
        "how can i safely remove body hair?",
        "is body hair removal necessary?"
    ], [
        "Some options for body hair removal include shaving, waxing, and laser hair removal. To safely remove body hair, it's important to use clean, sharp tools and follow proper techniques to avoid irritation and infection. Body hair removal is a personal choice and not necessary for everyone."
    ]],
    [[
        "what are some safe binding methods for trans men?",
        "how can i safely tuck as a trans woman?",
        "what are the risks of binding and tucking?"
    ], [
        "Some safe binding methods for trans men include using a binder or compression shirt designed for binding, and taking breaks to stretch and breathe. To safely tuck as a trans woman, it's important to use proper techniques and materials, such as a gaff or specialized underwear. The risks of binding and tucking include skin irritation, discomfort, and breathing difficulties if done incorrectly or for extended periods of time."
    ]],
    [[
        "how can i improve my mental health as an lgbtq person?",
        "what are some mental health resources for the lgbtq community?",
        "i'm struggling with my mental health as an lgbtq person, what should i do?"
    ], [
        "There are many ways to improve your mental health as an LGBTQ person, including finding a therapist who is affirming of your identity, seeking support from LGBTQ organizations, practicing self-care, and connecting with supportive friends and family.",
        "There are several mental health resources available for the LGBTQ community, such as The Trevor Project, which provides crisis intervention and suicide prevention services for LGBTQ young people, and the National Alliance on Mental Illness (NAMI), which offers resources and support groups for LGBTQ individuals.",
        "If you're struggling with your mental health as an LGBTQ person, it's important to seek support. Consider reaching out to a therapist or counselor who is knowledgeable about LGBTQ issues, or contacting an LGBTQ organization for resources and support."
    ]],
    [[
        "what are some common physical health concerns for lgbtq people?",
        "how can i stay healthy as an lgbtq person?",
        "what are some lgbtq-friendly healthcare providers?"
    ], [
        "LGBTQ people may face unique physical health concerns, such as higher rates of HIV/AIDS and certain cancers. It's important to prioritize preventative care and regular check-ups with a healthcare provider who is knowledgeable about LGBTQ health issues.",
        "To stay healthy as an LGBTQ person, make sure to prioritize self-care, exercise regularly, and eat a balanced diet. It's also important to stay up to date on routine screenings and preventative care.",
        "There are several resources available for finding LGBTQ-friendly healthcare providers, such as the Gay and Lesbian Medical Association's provider directory, and the Human Rights Campaign's Healthcare Equality Index."
    ]],
    [[
        "what are some resources for transgender healthcare?",
        "how can i find a trans-friendly doctor?",
        "what are some common health concerns for transgender people?"
    ], [
        "There are several resources available for transgender healthcare, such as the World Professional Association for Transgender Health's provider directory, and the Transgender Legal Defense and Education Fund's healthcare resources.",
        "To find a trans-friendly doctor, consider reaching out to local LGBTQ organizations or support groups for recommendations. It's also important to research potential providers and ask questions about their experience working with transgender patients.",
        "Transgender people may face unique health concerns, such as gender dysphoria and discrimination from healthcare providers. It's important to seek out providers who are knowledgeable about transgender health issues and affirming of your gender identity."
    ]]]


def send_message(message):
    response = respond(message)
    return response


def respond(message):
    should_query_be_done = True
    for key in range(0, len(patterns)):
        if message.lower() in patterns[key][0]:
            response = patterns[key][1][random.randint(0, len(patterns[key][1]) - 1)]
            should_query_be_done = False
            return response
            break;
        for text in patterns[key][0]:
            message_doc = nlp(message)
            text_doc = nlp(text)
            if message_doc and text_doc:
                similarity_score = message_doc.similarity(text_doc)
            if similarity_score >= 0.6:
                response = patterns[key][1][random.randint(0, len(patterns[key][1] - 1))]
                should_query_be_done = False
                return response
                break
    if should_query_be_done:
        query_search(message)


def community_or_event(doc):
    comm_presence = False
    loc_presence = False
    for ent in entity_list:
        if ent[0] in all_communities:
            comm_presence = True
            community = ent[0]
        if ent[1] == "GPE":
            loc_presence = True
            location = ent[0]
    if loc_presence:
        if comm_presence:
            DataBase.comm_query(location, community)
        else:
            DataBase.comm_query(location)
    elif (comm_presence):
        DataBase.comm_query(community)
    event_presence = False
    location_presence = False
    for ent in entity_list:
        if ent[0] in all_events:
            event_presence = True
            event = ent[0]
        if ent[1] == "GPE":
            location_presence = True
            location = ent[0]
    if location_presence:
        if event_presence:
            DataBase.event_query(location, event)
        else:
            DataBase.event_query(location)
    elif event_presence:
        DataBase.event_query(event)

    if (not comm_presence and not loc_presence) and (not event_presence and not location_presence):
        coh(message)


def query_search(message):
    doc = nlp(message)
    entity_list = [[ent.text, ent.label_] for ent in doc.ents]
    for ent in doc.ents:
        if (ent.text.lower() in gender or ent.text.lower() == 'gender') and re.search(
                r"(explain|describe|elaborate|tell|say|what|\.are|\. are|\.is |\.can|\. can|?|\.could|\. could|which|where|\.were|\. were )",
                message.lower()) is not None:
            DataBase.gender_query(ent.text)
            break
        if (ent.label_ == 'ORG' and re.search(
                r"(explain|describe|elaborate|tell|say|what|\.are|\. are|\.is |\.can|\. can|?|\.could|\. could|which|where|\.were|\. were  )",
                message.lower()) is not None) or (
                re.search(r"(community|group|events|celebration|venue|celebrate)", ent.text.lower()) is not None):
            community_or_event(doc)
            break


check=True
while(check):
        message = input("Let's chat \n")
        if message.lower()!='exit':
                send_message(message)
        else:
                print("Hope to see you again")
                check=False
def coh(msg):
    co = cohere.Client('')  # Enter your CoHere API key
    response_coh = co.generate(model='command', prompt=msg, max_tokens=300, temperature=0.9, k=0, stop_sequences=[],
                               return_likelihoods='NONE')
    response = ('Prediction: {}'.format(response_coh.generations[0].text))
    return response

