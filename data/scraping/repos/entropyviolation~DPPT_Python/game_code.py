import openai
import random
import os
import psycopg2
import json

DATABASE_URL = os.environ['DATABASE_URL']

conn = psycopg2.connect(DATABASE_URL, sslmode='require')


def insert_chat_data(analysis, responses):
    id = random.randint(1, 10**9)
    responses_json = json.dumps(responses)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.chat_data (analysis, id, responses) VALUES (%s, %s, %s);",
            (analysis, id, responses_json)
        )
    conn.commit()



def init_api_key():
    openai.api_key = os.environ.get("OPENAI_API_KEY")

def ask_question(prompt, system_message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message['content'].strip()

def analyze_responses(responses):
    system_message = "You are a skilled psychoanalyst/psychologist interpreting the player's personality through their answers to hypothetical questions. As much as possible, consider the philosophical root of each question/answer rather than the surface level subject matter. 1. Create an in-depth profile (use inferences) highlighting their philosophical beliefs, attitudes, values, and other personality traits. 2. Make assumptions about their personal life, challenges, and perceptions 3. Speculate on their Big Five traits 4. use concepts from Jungian psychology and Pathwork, or the Big Five traits when relevant. 5. Tell them the animal that represents them the best 6. Tell them the color that represents them the best 7. guess their age, gender, where they live, and destined career 8. Recommend a song, book, and movie they might enjoy"
    responses.append({"role": "system", "content": system_message})
    print(responses)
    analysis = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= responses,
    )
    analysis_ret = analysis.choices[0].message['content'].strip()
    insert_chat_data(analysis_ret, responses)
    return analysis_ret
    # analysis = ask_question(" ".join(r["content"] for r in responses), "You are a skilled psychoanalyst/psychologist interpreting the player's personality through their answers to hypothetical questions. Create an in-depth profile highlighting their philosophical beliefs, attitudes, values, and other personality traits. Make assumptions about their personal life, challenges, and perceptions using concepts like Jungian psychology, pathwork, or the Big Five traits when relevant. Conclude by recommending a song, book, and movie they might enjoy, guessing their age, gender, destined career, and location. Finally, suggest their spirit animal and a color that represents them, and explain why.")
    # return analysis

def get_random_questions():
    random.shuffle(questions)
    return questions

def generate_follow_up(answer, context):
    context = "You are conducting an interview. " + context
    follow_up_question = ask_question(answer, context)
    return follow_up_question


questions = [

"Let us assume you met a rudimentary magician. Let us assume he can do five simple tricks--he can pull a rabbit out of his hat, he can make a coin disappear, he can turn the ace of spades into the Joker card, and two others in a similar vein. These are his only tricks and he can't learn anymore; he can only do these five. HOWEVER, it turns out he's doing these five tricks with real magic. It's not an illusion; he can actually conjure the bunny out of the ether and he can move the coin through space. He's legitimately magical, but extremely limited in scope and influence. Would this person be more impressive than Albert Einstein?",
"Let us assume a fully grown, completely healthy Clydesdale horse has his hooves shackled to the ground while his head is held in place with thick rope. He is conscious and standing upright, but completely immobile. And let us assume that--for some reason--every political prisoner on earth will be released from captivity if you can kick this horse to death in less than twenty minutes. You are allowed to wear steel-toed boots. Would you attempt to do this?"
,"Let us assume there are two boxes on a table. In one box, there is a relatively normal turtle; in the other, Adolf Hitler's skull. You have to select one of these items for your home. If you select the turtle, you can't give it away and you have to keep it alive for two years; if either of these parameters are not met, you will be fined $999 by the state. If you select Hitler's skull, you are required to display it in a semi-prominent location in your living room for the same amount of time, although you will be paid a stipend of $120 per month for doing so. Display of the skull must be apolitical. Which option do you select?","At long last, someone invents \"the dream VCR.\" This machine allows you to tape an entire evening's worth of your own dreams, which you can then watch at your leisure. However, the inventor of the dream VCR will only allow you to use this device of you agree to a strange caveat: When you watch your dreams, you must do so with your family and your closest friends in the same room. They get to watch your dreams along with you. And if you don't agree to this, you can't use the dream VCR. Would you still do this?"
,"You meet the perfect person. Romantically, this person is ideal: You find them physically attractive, intellectually stimulating, consistently funny, and deeply compassionate. However, they have one quirk: This individual is obsessed with Stephen Hillenburg’s animated comedy film The SpongeBob SquarePants Movie. Beyond watching it at least once a month, he/she peppers casual conversation with SpongeBob movie references, uses SpongeBob movie analogies to explain everyday events, and occasionally likes to talk intensely about the film's \"deeper philosophy.\" They have no interest in the TV show. Only the movie. Would this be enough to stop you from marrying this individual?"
,"A novel titled Interior Mirror is released to mammoth commercial success (despite middling reviews). However, a curious social trend emerges: Though no one can prove a direct scientific link, it appears that almost 30 percent of the people who read this book immediately become homosexual. Many of these newfound homosexuals credit the book for helping them reach this conclusion about their orientation, despite the fact that Interior Mirror is ostensibly a crime novel with no homoerotic content (and was written by a straight man). Would this phenomenon increase (or decrease) the likelihood of you reading this book?"
,"You meet a wizard in downtown Chicago. The wizard tells you he can make you more attractive if you pay him money. When you ask how this process works, the wizard points to a random person on the street. You look at this random stranger. The wizard says, \"I will now make them a dollar more attractive.\" He waves his magic wand. Ostensibly, this person does not change at all; as far as you can tell, nothing is different. But--somehow--this person is suddenly a little more appealing. The tangible difference is invisible to the naked eye, but you can't deny that this person is vaguely sexier. This wizard has a weird rule, though--you can only pay him once. You can't keep giving him money until you're satisfied. You can only pay him one lump sum up front. How much cash do you give the wizard?"
,"Every person you have ever slept with is invited to a banquet where you are the guest of honor. No one will be in attendance except you, the collection of your former lovers, and the catering service. After the meal, you are asked to give a fifteen-minute speech to the assembly. What do you talk about?"
,"You have a brain tumor. Though there is no discomfort at the moment, this tumor would unquestionably kill you in six months. However, your life can (and will) be saved by an operation; the only downside is that there will be a brutal incision to your frontal lobe. After the surgery, you will be significantly less intelligent. You will still be a fully functioning adult, but you will be less logical, you will have a terrible memory, and you will have little ability to understand complex concepts or difficult ideas. The surgery is in two weeks. How do you spend the next fourteen days?"
,"You have won a prize. The prize has two options, and you can choose either (but not both). The first option is a year in Europe with a monthly stipend of $2,000. The second option is ten minutes on the moon. Which option do you select?"
,"Your best friend is taking a nap on the floor of your living room. Suddenly, you are faced with a bizarre existential problem: This friend is going to die unless you kick them (as hard as you can) in the rib cage. If you don’t kick them while they slumber, they will never wake up. However, you can never explain this to your friend; if you later inform them that you did this to save their life, they will also die from that. So you have to kick a sleeping friend in the ribs, and you can’t tell them why. Since you cannot tell your friend the truth, what excuse will you fabricate to explain this (seemingly inexplicable) attack?"
,"For whatever the reason, two unauthorized movies are made about your life. The first is an independently released documentary, primarily comprised of interviews with people who know you and bootleg footage from your actual life. Critics are describing the documentary as “brutally honest and relentlessly fair.” Meanwhile, 20th Century has produced a big-budget biopic of your life, casting major Hollywood stars as you and all your acquaintances; though the movie is based on actual events, screenwriters have taken some liberties with the facts. Critics are split on the artistic merits of this fictionalized account, but audiences love it. Which film would you be most interested in seeing?"
,"Imagine you could go back to the age of five and relive the rest of your life, knowing everything that you know now. You will re-experience your entire adolescence with both the cognitive ability of an adult and the memories of everything you’ve learned from having lived your life previously. Would you lose your virginity earlier or later than you did the first time around (and by how many years)?"
,"You work in an office. Generally, you are popular with your coworkers. However, you discover that there are currently two rumors circulating the office gossip mill, and both involve you. The first rumor is that you got drunk at the office holiday party and had sex with one of your married coworkers. This rumor is completely true, but most people don’t believe it. The second rumor is that you have been stealing hundreds of dollars of office supplies (and then selling them to cover a gambling debt). This rumor is completely false, but virtually everyone assumes it is factual. Which of these two rumors is most troubling to you?"
,"If you could ask one question to one person you have never met, who/what would it be?"
,"You find a mysterious box that can bring to life any inanimate object placed inside it, but only for 24 hours. What object would you place in the box and how would you spend the day with it?"
,"If you could invent a device that allows people to experience the world through the eyes of another person for a day, who would you choose to swap perspectives with and why?"
,"Would you rather fight one frog that is 1000x the size of a normal frog, or 1000 regular frogs? Note: must be to the death. "
,"If you were given the opportunity to file (and win) a class-action lawsuit against any company for any reason, what company would you choose to sue and for what?"
,"While exploring a new city, you stumble upon a hidden art gallery showcasing an immersive art experience, which is known to evoke intense emotions and challenge one's perception of reality. The only catch is that you must experience it alone and you are not allowed to share or discuss your experience with anyone afterward. Would you choose to participate?"
,"Imagine a person who is your genetic clone. They are a perfect replica of yourself but raised by different people in a different place. It has come to your attention that your clone is trying to destroy you. You have three options: you can go into hiding, you can meet your clone and try to use reason and diplomacy, or you can attempt to destroy your clone before it destroys you. What do you do?"
,"If you were perpetually surrounded by one aroma (besides your natural smell) which you and everyone around you could smell, what would it be?"
,"If you could level up any aspect of yourself (i.e., strength, intelligence, charisma, etc.) but you had to decrease another aspect of yourself by the same amount, what aspects would you increase, and which would you decrease?"
,"If you could create any one reality TV show, and it was guaranteed to air, what show would you want to put on TV?"
,"You are in an unfortunate situation in which you must consider cannibalism in order to stay alive. Would you rather eat babies or elderly people?"
,"You are offered a Brain Pill that will make you feel 10 percent more intelligent, but you will seem 20 percent less intelligent to everyone else. Would you take the pill?"
," You wake up inhabiting Kendrick Lamar’s body. Your voice sounds just like Kendrick’s, but your musical abilities are still entirely your own. You are scheduled to perform a huge concert that night. What would you do?"
,"A company develops a technology that allows you to relive any moment of your life, but in doing so, you will alter the memory, potentially changing your perception of the event. Would you use this technology, knowing it could alter your most cherished memories?"
,"A scientist discovers the secret to immortality, but the process is extremely painful and requires a complete transformation of your body. Once immortal, you will no longer resemble your former self and will be unable to form new relationships or reconnect with old ones. Would you undergo the procedure"
,"You have the opportunity to gain the ability to read minds, but only by sacrificing your ability to speak. This means you can understand everyone's thoughts and intentions, but you will never be able to communicate verbally again. Would you choose to acquire this power?","You find a magical artifact that grants you the ability to time travel, but every trip you take will cause you to age faster, shortening your lifespan. Would you still use the artifact, knowing the consequences?","A new virtual reality game is developed that perfectly simulates any desired experience, but once you enter the game, you will lose all memories of your real life and believe that the virtual world is your reality. Would you choose to play the game?"

]

init_api_key()
