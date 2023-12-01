from app.models import db, Post

def seed_posts():
    post_1 = Post(
        title="[AskJS] Are leetcodes good interview questions?",
        content="""Do you think leetcode-style algo problems are good interview questions? Just in case, here are some examples that I have in mind:

1. Count unique values in an array.
2. Given a linked list, return true iff it has a loop.
3. Implement a data structure that stores numbers with O(1) insertion and removal, and O(log n) search.

Bonus: if your answer is "yes", how do you tailor interviews to junior / middle / senior positions?""",
        user_id=13,
        community_id=3
    )
    post_2 = Post(
        title="Mostly adequate guide to functional programming (in JavaScript)",
        link_url="https://mostly-adequate.gitbook.io/mostly-adequate-guide/",
        user_id=14,
        community_id=3
    )
    post_3 = Post(
        title="Anyone want a tiger?",
        content="I'm selling my tiger for 50 gold. Reminder that you must be level 30 or higher to be able to buy it. Message me if interested.",
        user_id=3,
        community_id=1,
    )
    post_4 = Post(
        title="This flawless yellow onion",
        img_url="https://i.redd.it/bpq28osspxda1.jpg",
        user_id=4,
        community_id=4
    )
    post_5 = Post(
        title="Any idea what kind of cat I just adopted? [oc]",
        img_url = "https://i.redd.it/7giweloj1uda1.jpg",
        user_id = 5,
        community_id = 5
    )
    post_6 = Post(
        title="Of the 69 things they tested me for, I'm allergic to 60 of them.",
        img_url = "https://i.redd.it/a3wj960mpwda1.jpg",
        user_id = 6,
        community_id = 6
    )
    post_7 = Post(
        title="Accomplishments",
        img_url = "https://i.redd.it/50fhg3lhmzda1.jpg",
        user_id = 7,
        community_id = 7
    )
    post_8 = Post(
        title="YSK Overdosing on Tylenol (Acetaminophen/Paracetamol/Panadol) is a slow, painful way to die",
        content='''Why YSK: Due to being cheap, readily available, and easily accessible, Tylenol overdose has become one of the most common suicide methods. However, people don’t realize how truly awful a Tylenol overdose really is.

        The first 0-24 hours you may feel nothing, or mild symptoms. After that, the torture begins. You will suffer profuse vomiting, edema (your body swells up), a crushing headache, turn yellow, and feel like your liver and kidneys are being stabbed nonstop, what people often describe as the worst pain they’ve ever experienced, and bleeding. This goes on for days or even weeks. Meanwhile, if you reach the final stage, you are suffering as you wait for a liver transplant or death (when your organs shut down).

        Most people survive Tylenol overdoses; however, they are often left with permanent liver/kidney damage, which can require things such as dialysis for multiple hours several times per week, medication, and lifestyle changes.

        What can be done? 3 things:

        Educate people on the reality of an OD.

        Mandate that Tylenol be sold in blister-pack form. When the UK implemented this, intentional overdoses declined by 43 percent!

        Treat the underlying problem. We have a mental health crisis, and it’s not going to go away on its own. Developed countries especially need to work on providing resources to people with mental health conditions, before it escalates to suicide.

        Source: https://www.merckmanuals.com/home/injuries-and-poisoning/poisoning/acetaminophen-poisoning''',
        user_id = 8,
        community_id = 8
    )
    post_9 = Post(
        title="Fellas what are some subtle signs a women is “toxic”?",
        content="",
        user_id = 9,
        community_id = 9
    )
    post_10 = Post(
        title="Any idea what kind of cat I just adopted? [OC]",
        img_url = "https://i.redd.it/26r2fc2miyda1.jpg",
        user_id = 10,
        community_id = 5
    )
    post_11 = Post(
        title="These rice fields shaped like spiderwebs at Indonesia",
        img_url = "https://i.redd.it/ipedxxzg4tda1.jpg",
        user_id = 11,
        community_id = 4
    )
    post_12 = Post(
        title="They used this picture at work today to see if we could locate all the OSHA safety violations...",
        img_url="https://preview.redd.it/4xu2rr4w7oia1.png?width=960&crop=smart&auto=webp&v=enabled&s=4fdf879e34a95c2ac6908489285cc3d2a1fbbb8f",
        user_id=2,
        community_id=10
    )
    post_13 = Post(
        title="My social security was canceled",
        img_url="https://i.redd.it/osfw2m6zamia1.jpg",
        user_id=2,
        community_id=10
    )
    post_14 = Post(
        title="Got a vasectomy today. My wife got me get well cookies.",
        img_url="https://external-preview.redd.it/E7K1PbUhf7giIAhUgeoUKuiuccD9TBQtkh2m8HBfOjs.jpg?auto=webp&v=enabled&s=5cd90ba4f73d5a00b896ec929f57fab41884a4cc",
        user_id=3,
        community_id=10
    )
    post_15 = Post(
        title="BRAILLE?!?!",
        img_url="https://preview.redd.it/oo5glv19jmia1.jpg?width=640&crop=smart&auto=webp&v=enabled&s=6edac192344ff2d3d90f3efb17eda9755afa87d0",
        user_id=4,
        community_id=10
    )
    post_16=Post(
        title="API key scraping?",
        content="""I made an AI image generator using OpenAI, but when I pushed it to github, I forgot that I'd put another instance of the key in (in a header that I'd then forgotten I'd done, and didn't use .env), wasn't a huge issue as it was just for personal learning.

Anyway, I immediately got an email from Openai to tell me the key had been leaked and they'd given me a new API key. My question is, how did they know?

Are they constantly scraping the Web for api keys?
        """,
        user_id=5,
        community_id=11
    )
    post_17=Post(
        title="How to borrow other people's code?",
        content="""Hello,

So I am doing my own web app and I saw some code in GitHub that would help me. The code is licensed as MIT. Do I just use the code and put a comment linking to the source? Do I need to get the License file from the repo and put it in mine too? If so where does the License File go? Perhaps there is file like a README.md that mentions where you got the code?

I never needed to borrow code before so I ain't sure how this works. Sorry for my bad English.

Thanks in advance for your help.""",
        user_id=6,
        community_id=11
    )
    post_18=Post(
        title="Ignoring me for a week, is this good enough?",
        img_url="https://preview.redd.it/ekcf3y3njuba1.jpg?width=960&crop=smart&auto=webp&v=enabled&s=7e0a3dfb41d91f5bd68891f2fb0b040c4084d5cc",
        user_id=6,
        community_id=11
    )
    post_19=Post(
        title="Cheesecake Goes to the Vet",
        img_url="https://i.redd.it/bf2m09a9mer91.jpg",
        user_id=17,
        community_id=1
    )
    post_20=Post(
        title="This made me smile so I hope it will make you smile too",
        img_url="https://i.redd.it/9314muue51y81.jpg",
        user_id=24,
        community_id=1
    )
    post_21=Post(
        title="Cats Cats Cats!!!",
        img_url="https://i.redd.it/nyw9d2ggam391.png",
        user_id=21,
        community_id=1
    )
    post_22=Post(
        title="🤓🧠",
        img_url="https://i.redd.it/6at2tb8b8w591.jpg",
        user_id=13,
        community_id=2
    )
    post_23=Post(
        title="Quite a lesson indeed",
        img_url="https://i.redd.it/z730z4dck3i91.jpg",
        user_id=3,
        community_id=2
    )
    post_24=Post(
        title="core-js maintainer: “So, what’s next?”",
        link_url="https://github.com/zloirock/core-js/blob/master/docs/2023-02-14-so-whats-next.md",
        user_id=2,
        community_id=3
    )
    post_25=Post(
        title="\"Dev burnout drastically decreases when you actually ship things regularly. Burnout is caused by crap like toil, rework and spending too much mental energy on bottlenecks.\" Cool conversation with the head engineer of Slack on how burnout is caused by all the things that keep devs from coding.",
        link_url="https://devinterrupted.substack.com/p/the-best-solution-to-burnout-weve",
        user_id=2,
        community_id=3
    )
    post_26=Post(
        title="JavaScript is the Most Demanded Programming Language in 2022, 1 out of 3 dev jobs require JavaScript knowledge.",
        link_url="https://www.devjobsscanner.com/blog/top-8-most-demanded-languages-in-2022/",
        user_id=7,
        community_id=3
    )
    post_27=Post(
        title="[AskJS] Are there people using vanilla JS? If so: What are you doing with it?",
        content="""Everything seems to be about APIs, frameworks, libraries and installable serverside modules nd compiling static sites before deploying them to the web.

But are there still people using vanilla JS? The oldschool stuff that simply runs in the browser by adding <code><script src="....."></code> to the HTML's head area without having to set up a "full stack node react latest-buzzword development environment"?

So, all vanilla JS users: What are you using JS for?

For a private project I wrote a custom client-side router and some asynchronous fetching and displaying functions with some logic to format stuff as needed. Total files amount including the <code>index.html</code>: 3. All fully flexible and customizable as needed for the specific project, zero overhead.""",
        user_id=11,
        community_id=3
    )
    post_28=Post(
        title="Saturn through my 6\" telescope",
        img_url="https://i.redd.it/dfooeoh9lpw81.jpg",
        user_id=5,
        community_id=12
    )
    post_29=Post(
        title="TIL in the early 90s LL Cool J shared with his grandma that he couldn't survive as a rapper now that gangsta rap was popular. His grandma responded, \"Oh baby, just knock them out!\" which inspired him to write 'Mama Said Knock You Out' a grammy award winning certified platnum single.",
        link_url="https://en.wikipedia.org/wiki/Mama_Said_Knock_You_Out_(song)",
        user_id=10,
        community_id=13
    )
    post_30=Post(
        title="This is what hanging out in a college dorm room looked like in 1910. (University of Illinois)",
        img_url="https://i.redd.it/zjkqzqgnpa491.jpg",
        user_id=15,
        community_id=14
    )
    post_31=Post(
        title="Player got kicked from a professional esports team because his mom was in the final stages of her cancer.",
        img_url="https://i.redd.it/96ddaelzxm091.jpg",
        user_id=12,
        community_id=15
    )
    post_32=Post(
        title="A Message to All the Self Taught Devs Feeling Discouraged...",
        content="""...by the portfolio's that intimidate you...

...by the posts that reference technologies you dont even understand...

...by the fear you'll never pass an interview despite all the things you've built...

...by all the worry in your mind that you can learn so much online but everyone else seems to constantly be "better" than you...

...Dont be.

This is your story. Your journey to walk. You ARE learning. Every day. Even when you arent coding, you are cemeting and committing that knowledge to memory. Remember the first time you learned a <div>?

And now you probably write them all the time, without thinking. Look at your growth. That will be the same with all your React Apps, and your career.

You are growing. Keep at it. You'll get there. Just dont give up and ask for help whenever you need it! And never discount yourself for all the work youve already done. Happy Coding ❤""",
        user_id=2,
        community_id=16
    )

    post_33=Post(
        title="I have made a list of the best Flask tutorials for beginners to learn web development. Beginners will benefit from it.",
        link_url="https://medium.com/quick-code/top-online-tutorials-to-learn-flask-python-c2723df5326c",
        user_id=16,
        community_id=17
    )

    post_34=Post(
        title="Lad wrote a Python script to download Alexa voice recordings, he didn't expect this email.",
        img_url="https://i.redd.it/2s0dj8ob12u41.png",
        user_id=2,
        community_id=18
    )

    post_35=Post(
        title="I solved a real life problem with python for the 1st time and I feel like a wizard",
        content="""Okay, this is probably going to sound super dumb, but today I'm putting the finishing touches on a program that downloads a data file for me (into a folder of my choosing!!! this part tripped me up for a while) and renames it according to today's date and I feel like a goddamn SORCERESS.

I showed it to my boyfriend, and then I felt kinda sheepish, because like, okay, it's just a file but .... it's so incredible for me to just see it working!! I know it sounds simple, but I had to navigate around so many barriers to make it work, and now that I've mastered this, so many other amazing projects feel accessible and understandable to me.

I'm just so happy! It feels like all my hard work teaching myself this stuff has paid off. Just wanted to share with you guys. :)""",
        user_id=6,
        community_id=19
    )

    post_36=Post(
        title="We ordered a grill. Got 300 iPads",
        img_url="https://i.redd.it/5ouf4y09w5v81.jpg",
        user_id=50,
        community_id=20
    )

    post_37=Post(
        title="Any other visual learners out there? I came across this and thought it was helpful.",
        img_url="https://i.redd.it/1fkv59wp0dc91.jpg",
        user_id=7,
        community_id=21
    )

    post_38=Post(
        title="Had a day off due to Covid and did this for my wife",
        img_url="https://i.redd.it/mwrvyrplret41.jpg",
        user_id=45,
        community_id=4
    )

    post_39=Post(
        title="How a Mongolian dresses their child for the cold",
        img_url="https://i.redd.it/ote3bpr3lsma1.jpg",
        user_id=39,
        community_id=5
    )

    post_40=Post(
        title="My Doordash driver attached this to my order.",
        img_url="https://i.redd.it/b3jvbmenansa1.png",
        user_id=13,
        community_id=6
    )
    post_41=Post(
        title="Why aren’t you playing by the rules of the game!",
        img_url="https://i.redd.it/bq54d2yu1ss91.jpg",
        user_id=8,
        community_id=7
    )
    post_42=Post(
        title="YSK that Harvard offers a free certificate for its Intro to Computer Science & Programming Education",
        content="""Why YSK: Harvard is one of the world's top universities. But it's very expensive and selective. So very few people get to enjoy the education they offer.

However, they've made CS50, Harvard's Introduction to Computer Science and Programming, available online for free. And upon completion, you even get a free certificate from Harvard.

I can't overstate how good the course is. The professor is super engaging. The lectures are recorded annually, so the curriculum is always up to date. And it's very interactive, with weekly assignments that you complete through an in-browser code editor.

To top it all off, once you complete the course, you get a free certificate of completion from Harvard. Very few online courses offer free certificates nowadays, especially from top universities.

You can take the course for free on Harvard OpenCourseWare:

https://cs50.harvard.edu/x/2022/

(Note that you can also take it through edX, but there, the certificate costs $150. On Harvard OpenCourseWare, the course is exactly the same, but the certificate is entirely free.)

I hope this help.""",
        user_id=17,
        community_id=8
    )
    post_43=Post(
        title="What are things women think men care about that you guys actually dont?",
        content="Girl here lmfao. Im just wondering what are some things were super self conscious about or like we worry it will be a deal breaker for you guys that u guys actually dont care about at all. I hope this makes sense sorry.",
        user_id=49,
        community_id=9
    )
    post_44=Post(
        title="My dad wrote to JRR Tolkien in 1959. Tolkien sent him a letter back.",
        img_url="https://i.redd.it/dt63tjlyvisa1.jpg",
        user_id=38,
        community_id=6
    )
    post_45=Post(
        title="Frosted class effect in CSS - one of my favorites",
        img_url="https://i.redd.it/w4wljb4t23u81.jpg",
        user_id=8,
        community_id=11
    )
    post_46=Post(
        title="Boston moved it’s highway underground in 2003. This was the result.",
        img_url="https://i.redd.it/egsjepstjrv81.jpg",
        user_id=30,
        community_id=12
    )
    post_47=Post(
        title="When Junior Developer asked Senior developer for code review.",
        img_url="https://i.redd.it/7tromt77jr991.png",
        user_id=22,
        community_id=21
    )
    post_48=Post(
        title="Kids solve all problems",
        img_url="https://i.redd.it/7czr2yswh0291.jpg",
        user_id=34,
        community_id=20
    )
    post_49=Post(
        title="Got to say it, IMO, the book 'Python Crash Course', is far superior for a beginner than 'Automate the Boring Stuff'",
        content="""I read upto lists and dictionaries in Automate the Boring stuff, and watched the videos on youtube for those chapters. The excercises seemed to ask for stuff that i had not learnt or were far ahead of my learning so far.

Dived into 'Python Crash Course' and haven't looked back. This book is fun, engaging, and all the excersises are relevant to what you have just learnt.

I will go back to 'Automate' but was overwhelmed and skipped most of the chapter excercises, as they seemed too difficult""",
        user_id=23,
        community_id=19
    )

    post_50=Post(
        title="Built a flask web app! It’s a self hosted music server that can cache, organise and play youtube tracks.",
        img_url="https://i.redd.it/hy1yrn9szb751.jpg",
        user_id=18,
        community_id=18
    )
    post_51=Post(
        title="I redesign the Python logo to make it more modern",
        img_url="https://i.redd.it/rxezjyf4ojx41.png",
        user_id=6,
        community_id=17
    )
    post_52=Post(
        title="Apparently submitting assignments before the due date is considered “Late”.",
        img_url="https://i.redd.it/d3fiudd6a4ga1.jpg",
        user_id=20,
        community_id=15
    )
    post_53=Post(
        title="TIL in 2011, a 29-year-old Australian bartender found an ATM glitch that allowed him to withdraw way beyond his balance. In a bender that lasted four-and-half months, he managed to spend around $1.6 million of the bank’s money.",
        link_url="https://www.vice.com/en/article/pa5kgg/this-australian-bartender-dan-saunders-found-an-atm-bank-glitch-hack-and-blew-16-million-dollars?utm_source=reddit.com",
        user_id=5,
        community_id=13
    )
    post_54=Post(
        title="I made a list of 70+ open-source clones of sites like Airbnb, Tiktok, Netflix, Spotify etc. See their code, demo, tech stack, & github stars.",
        content="""I curated a list of 70+ open-source clones of popular sites like Airbnb, Amazon, Instagram, Netflix, Tiktok, Spotify, Trello, Whatsapp, Youtube, etc. List contains source code, demo links, tech stack, and, GitHub stars count. Great for learning purpose!

More open-source contributions are welcome to grow this list.

GitHub link: https://github.com/GorvGoyl/Clone-Wars

Pretty view: https://gourav.io/clone-wars

I was building this list for a while... Please share it with others 🙏""",
        user_id=11,
        community_id=16
    )
    post_55=Post(
        title="Why is the difference between Mothers and Fathers Day so astounding?",
        content="""Commercials, lunches, etc. If my sister didn't get a gift on mother's day she'd bitch. She bitched that no one did enough as is.

Today I didn't even get a text from her.

Men do get the short end. I had lunch with my son but it was like pulling teeth and I had to pay.""",
        user_id=45,
        community_id=9
    )
    post_56=Post(
        title="YSK if you have a minimum wage job, the employer cannot deduct money from checks for uniforms, missing cash, stolen meals, wrong deliveries, damaged products, etc. You absolutely have to get paid a minimum wage.",
        content="""Why YSK: It's extremely common for employers to deduct losses from employee's checks if they believe the employee had some responsibility for that loss. In some states this is illegal as well, but overall the employer cannot do this if it means you will earn less than minimum wage.

Some states enacted laws that force employers to pay out triple damages for violations of several wage laws. Most states will fine the company $1000.

https://www.epi.org/publication/employers-steal-billions-from-workers-paychecks-each-year/""",
        user_id=42,
        community_id=8
    )
    post_57=Post(
        title="18-year-old crowned Miss America Ruth Malcolmson. 1924",
        img_url="https://i.redd.it/51eu19uv3ts91.jpg",
        user_id=20,
        community_id=14
    )
    post_58=Post(
        title="Seems legit?",
        img_url="https://i.redd.it/yn1guwwbj6991.jpg",
        user_id=25,
        community_id=20
    )
    post_59=Post(
        title="I keep forgetting this stuff so I compiled my notes for building Flask APIs into a short guide",
        link_url="https://www.mvanga.com/guides/concise-guide-to-building-flask-apis",
        user_id=26,
        community_id=17
    )

    post_60=Post(
        title="Aerial Picture of an uncontacted Amazon Tribe",
        img_url="https://i.redd.it/4x4n4o82h6e91.jpg",
        user_id=35,
        community_id=12
    )
    post_61=Post(
        title="Grandma Was A Punk (1977)",
        img_url="https://i.redd.it/6r871gf7f0da1.jpg",
        user_id=30,
        community_id=14
    )
    post_62=Post(
        title="Debugging Cheat Sheet",
        img_url="https://i.redd.it/p1i8awsivji51.jpg",
        user_id=16,
        community_id=18
    )
    post_63=Post(
        title="when you take the job before reading the job description",
        img_url="https://i.redd.it/4fpwt2s4xw191.jpg",
        user_id=24,
        community_id=20
    )
    post_64=Post(
        title="TIL after being scolded by a woman who felt that his shoes were too expensive for kids, Shaq forwent a $40 million deal with Reebok & signed one with Walmart. He then brought in designers from Reebok so that his Walmart shoes would look costlier than the $20 price. Over 400 million pairs were sold.",
        link_url="https://sports.yahoo.com/shaq-reveals-why-rejected-40-130148701.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cucmVkZGl0LmNvbS8&guce_referrer_sig=AQAAAGGGKh0efIjXQMp6lGarS4xCHvB2Uq71EumxKgHHRMvCh5TwJDoZx1ExuMc139HAKeGeKRK3xEeaBBhZIwAPXV074eoSrD8FQV0cgQWasLlOogVKBfl_XjkXXyPpi-XF8GoYIRgnbNym9aLcSVRxN1A0v0wBAfMsaQOo8hYYBfIf",
        user_id=15,
        community_id=13
    )
    post_65=Post(
        title="Just found this site \"useHooks.com\" - super helpful collection of react hooks!",
        link_url="https://usehooks.com/",
        user_id=18,
        community_id=16
    )
    post_66=Post(
        title="For those of you struggling to think of a project",
        content="I'm not sure if this has been posted on here before, but if you are new and struggling to find a project to work on, go on Upwork/Fiverr and look at what people are trying to find freelancers for. Some of those listings will be for things you aren't capable of, but it will help you develop your skills.",
        user_id=48,
        community_id=19
    )
    post_67=Post(
        title="it's the most important skill",
        img_url="https://i.redd.it/g37eb3qivuv81.jpg",
        user_id=19,
        community_id=7
    )
    post_68=Post(
        title="Javascript Array Cheat Sheet with return types",
        img_url="https://i.redd.it/j1cpu57cr7r91.png",
        user_id=26,
        community_id=21
    )
    post_69=Post(
        title="My SO throws her daily contacts behind the headboard of our bed.",
        img_url="https://i.redd.it/dd7x7o9li8ha1.jpg",
        user_id=28,
        community_id=15
    )

    post_70=Post(
        title="This satisfying pebble I found at the beach.",
        img_url="https://i.imgur.com/HrfOXV0.jpeg",
        user_id=37,
        community_id=4
    )

    post_71=Post(
        title="My boyfriend tucks his dog in bed every night",
        content="That's basically it. I just wanted y'all to know that my boyfriend, who on the outside looks very stone cold and masculine, tucks his little dog in his bed every night, making sure his blanket is alright, places a toy next to him and repeatedly kisses him goodnight.",
        user_id=46,
        community_id=22
    )

    post_72=Post(
        title="Is it normal for kids to compliment me all the time?",
        content="I work in a kindergarten and have only been working here for a week and I think I get complimented by the kids constantly everyday. And about some weird things like \"I like your voice\" or \"you're the prettiest teacher ever\" or like \"I like your shoes\" or just straight up telling me they love me and i'm just really surprised at how affectionate little kids can be. i almost get no compliments in my day to day life so it's very surprising",
        user_id=6,
        community_id=22
    )

    post_73=Post(
        title="How I interview for FE positions",
        img_url="https://i.redd.it/1lg4xebu2hsa1.jpg",
        user_id=2,
        community_id=11
    )

    post_74=Post(
        title="Dockerfile explained: This Dockerfile creates a Docker image for a Flask app.",
        img_url="https://i.redd.it/35g8dixdgwz81.png",
        user_id=10,
        community_id=17
    )

    post_75=Post(
        title="Remember when Kenya tried to send 14 cows to America after 9/11?",
        content="That’s the cutest fucking thing. Like that’s absolutely adorable and we appreciate their act of kindness. Is there any other instances in history where countries tried to help each other like that? I know Mexico sent tortillas and Marines to aid with hurricane Katrina.",
        user_id=11,
        community_id=22
    )


    db.session.add(post_1)
    db.session.add(post_2)
    db.session.add(post_3)
    db.session.add(post_4)
    db.session.add(post_5)
    db.session.add(post_6)
    db.session.add(post_7)
    db.session.add(post_8)
    db.session.add(post_9)
    db.session.add(post_10)
    db.session.add(post_11)
    db.session.add(post_12)
    db.session.add(post_13)
    db.session.add(post_14)
    db.session.add(post_15)
    db.session.add(post_16)
    db.session.add(post_17)
    db.session.add(post_18)
    db.session.add(post_19)
    db.session.add(post_20)
    db.session.add(post_21)
    db.session.add(post_22)
    db.session.add(post_23)
    db.session.add(post_24)
    db.session.add(post_25)
    db.session.add(post_26)
    db.session.add(post_27)
    db.session.add(post_28)
    db.session.add(post_29)
    db.session.add(post_30)
    # db.session.add(post_31)
    # db.session.add(post_32)
    # db.session.add(post_33)
    # db.session.add(post_34)
    # db.session.add(post_35)
    # db.session.add(post_36)
    # db.session.add(post_37)
    # db.session.add(post_38)
    # db.session.add(post_39)
    # db.session.add(post_40)
    # db.session.add(post_41)
    # db.session.add(post_42)
    # db.session.add(post_43)
    # db.session.add(post_44)
    # db.session.add(post_45)
    # db.session.add(post_46)
    # db.session.add(post_47)
    # db.session.add(post_48)
    # db.session.add(post_49)
    # db.session.add(post_50)
    # db.session.add(post_51)
    # db.session.add(post_52)
    # db.session.add(post_53)
    # db.session.add(post_54)
    # db.session.add(post_55)
    # db.session.add(post_56)
    # db.session.add(post_57)
    # db.session.add(post_58)
    # db.session.add(post_59)
    # db.session.add(post_60)
    # db.session.add(post_61)
    # db.session.add(post_62)
    # db.session.add(post_63)
    # db.session.add(post_64)
    # db.session.add(post_65)
    # db.session.add(post_66)
    # db.session.add(post_67)
    # db.session.add(post_68)
    # db.session.add(post_69)
    # db.session.add(post_70)
    db.session.add(post_71)
    db.session.add(post_72)
    # db.session.add(post_73)
    # db.session.add(post_74)
    db.session.add(post_75)

    db.session.commit()

def undo_posts():
    db.session.execute("DELETE FROM posts")
    db.session.commit()
