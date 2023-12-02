from cohere.classify import Example

import csv

examples = [
    Example("Apple officially fined $19 million for not including chargers with iPhones", "negative"),
    Example("More #iOS16 bugs. It's getting ridiculous now", "negative"),
    Example("#Brazil Court Fines #Apple $19 million for shipping iPhones without chargers.", "negative"),
    Example("#Elon Musk stabs Ukraine in the back and demands that the Pentagon pay for the Starlink communication network helping fight Russia — or else SpaceX will pull the plug on the service, helping Putin in the process.", "negative"),
    Example("Elon Musk is trying to shake down the pentagon while pretending to be a hero.", "negative"),
    Example("Google is Racist", "negative"),
    Example(" Google, Meta & Amazon CAUGHT using fraudulent lobbying operations in an attempt to prevent the Digital Markets/Services Act from being passed, but it did, and is launching November 1st", "negative"),
    Example("why is netflix so fucking expensive in india? its almost 4 times more than amazon prime and you get music, books, free deliveries with prime", "negative"),
    Example("Bob, Amazon is a blood money business. If you take it serious, Your life will never remain the same", "negative"),
    Example("Dear @amazon, Stop advertising the sold out pink Spidey hoodie to me. Your targeted advertising is spot on. My daughter would love it. But if it’s sold out every time I click, you’re just making me angry.", "negative"),
    Example("Why the fuck is Rogers wifi always fucking down @RogersHelps", "negative"),
    Example(
        "HEY @Rogers BRING MY WIFI BACK I WAS IN THE MIDDLE OF A GAME??????", "negative"),
    Example("@Rogers MY WIFI IS DOWN WTF IS GOING ON YOU JUST COSTED MY RECORD IN FIFA YOU DISGRACE", "negative"),
    Example("The message for @RBC was loud and clear: Continuing on this climate-destroying path that violates Indigenous rights, will cost them their reputation and bottom line.", "negative"),
    Example("ok so i just ordered my xbox one power supply (microsoft sucks ass and i hate them i will never own a single one of there products again)", "negative"),
    Example("Microsoft Edge sucks Microsoft Insights more garbage 365 is puke invasive intrusive putrid SOMEONE HACK THESE COMMIES", "negative"),
    Example("FUCK YOU SAMSUNG FRIDGE!  #VIRTUALREDBAN", "negative"),
    Example("'I don't like Android because it is slow and lags'", "negative"),
    Example("I’m beyond disappointed @Apple I’ve only used 265GB of my storage and phone is slow af.", "negative"),



    Example("Battery life of my iPhone 13 Pro improved significantly after this iOS 16.0.3 update! Anyone else experiencing the same?", "positive"),
    Example("An iPhone 14 Pro sent the message to Police, was designed for a car crash while the owner was on a roller coaster.", "positive"),
    Example("Thank you #Google for introducing me to this course.", "positive"),
    Example("I am happy to share that I am starting a new position as Mobile Software Developer at TD!", "positive"),
    Example("Google shows me, my lover, from time to time.", "positive"),
    Example("I am grateful for the food that you send through Amazon", "positive"),
    Example("Having created this official illustration for Adobe/Amazon prior to seeing the show, I couldnt have known just how much I’d love Elrond! Can’t wait to rewatch this season.", "positive"),
    Example("This isnt breaking news, but body brushing is literally one of the best things you can do for your body, inside and out! So inexpensive, £1 you can get these brushes for, or from Amazon like where I got mine. Feels amazing. ", "positive"),
    Example("Ah! Always love it when Amazon recommends my own book to me. Feels like my book is going places!", "positive"),
    Example("Dealings in the Dark has been released for two full weeks! Have you picked up your copy? It's available on KU, as an ebook, as a paperback, and as a hardcover on Amazon! Pick up your copy today!", "positive"),
    Example("this school wifi is actually fast lmao it was at like 60mb/s download", "positive"),
    Example("I love that part of the Microsoft Security Score for Identity in Azure improves your score if you *don't* enforce password rotation, what a sign of the times!", "positive"),
    Example("I love Samsung Internet and Microsoft Edge a lot more than Chrome, but I keep using Chrome because that’s what I’m used to.", "positive"),
    Example("Microsoft has added a much better power options screen to the Xbox dashboard. It shows you how much power is used in both modes ", "positive"),
    Example("The IBM tools, resources and industry news you need to get the most value from your #data and make it truly #AI-ready.", "positive"),
    Example("In terms of display this Samsung Galaxy S23 is fitted with a 6.1-inch, 1,500 nits screen, while the design is referred to as 'sleek' and 'beautiful'. The design includes a hole-punch selfie camera, which is mounted in the top center of the screen.", "positive"),
    Example("One reason I'm a Samsung fan is because their trade in deals are on point, even outside the pre-order period. I know not everyone has something to trade in, but when you do, you really can't complain about price. Certainly beats Apple trade in process.", "positive"),
    Example("Inspiration, entertainment and innovation from Walmart and Sam's associates around the world | Home of Walmart and Sam's Radio #TeamWalmart #TeamSamsClub", "positive"),
    Example("The CHEAP BUT AWESOME tent is back in stock and on our Walmart registry!! These things are small and mighty, perfect to give out at City Hall, which has become a centralized spot for displaced folks this week! Thanks to everyone out there.", "positive"),
    Example("#SaitamaWolfPack iOS users that don’t want to wait for App Store updates, check out the 8” Galaxy TAB A7Lite at Walmart. I just bought it for $116, after tax.  It’s WiFi only. But it’ll only be used at home for SaitaPro. Cheap and will do the job. #Saitama #SaitaPro #UseSaitaPro", "positive"),
    Example("Microsoft Good work- life- balance and well paid", "positive"),


]

# with open('./csv/Tweets.csv') as file:
#     # python -m flask run
#     reader = csv.DictReader(file)

#     for row in reader:
#         # row is of type dict, with the following keys
#         # textID,text,selected_text,sentiment
#         text = row['text']
#         sentiment = row['sentiment']

#         if sentiment == "positive" or sentiment == "negative":
#             examples.append(Example(text, sentiment))
