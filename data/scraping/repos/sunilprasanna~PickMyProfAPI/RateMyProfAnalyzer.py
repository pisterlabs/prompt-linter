import sys
import praw
import cohere
from cohere.classify import Example

co = cohere.Client("XlPEiOs1Dnm83Avq8Gup5uy4tEjoYBTZO0IqcFBK")

reddit = praw.Reddit(client_id='CeFkFoVL5zNIoOC9u98M1Q',
                     client_secret='o5dMn2B7ffr1--UU_-qXsLgZ5IQ-LA',
                     password="Greninja#3005",
                     user_agent="testscript by u/fakebot3",
                     username="CosmicLatios")
subreddit = reddit.subreddit("utdallas")

cohereExamples = [
    Example("Edit: also I had him for D's&a not automata. Idk how his automata class isHw wasn't hard. Online exams were stupid. The content wasn't difficult but he basically structured them to where you had to speed run writing tables and stuff out in the exact format he wanted for no real reason. He purposely made it so we'd have to write out a large number of iterations on all of the algos so that we'd barely be able to finish the exam on time and therefore 'not cheat'. Even if you understood the algorithm correctly, wrote almost everything out, but maybe skipped writing a number or something in between calculations to save time (because no one could complete his exams on time) he'd take off almost all the points on the problem.It was incredibly dumb, and he ended up having to curve the entire class up like two letter grades bc of how poorly everyone was doingMaybe in person is better now :)", "negative"),
    Example("Yea ontop of everything he's just an ass as a person. Plus, one time during class he had the audacity to brag about how hard he had worked on his copy pasted from Google power point slides. That was when I lost it all. Man is legit insane.", "negative"),
    Example("The course is very easy and almost a free grade. It’s not modeling in Blender or Maya, or creating simulations in OpenGL or Unity, it’s modeling everything as something else. Like modeling the bookshelves in a library into an array.The class was basically a power point presentation every other week about how you turned a recipe into a series of if statements and for loops.There is a little bit of coding at the end of the semester but it’s very free form and open ended.", "positive"),
    Example("Automata is a hard class in general. I took Pervin for it last semester. You can expect a homework assignment every week over the lectures. He constantly says this should take you 5 minutes when in reality its closer to 1-2 hours. He had 3 tests throughout the semester that were very hard. There is no multiple choice/true false questions. In addition, Pervin has his own ways of doing things which makes YouTube completely irrelevant. He even says in his assignments to do things HIS way. He is a very old man who talks very slow. I was able to watch the lectures in 2x speed without problems. He did curve every exam pretty heavily. Overall I wouldn't recommend him but from others replies in this thread, Gupta doesn't sound great either. Typical UTD professor selection :D", "negative"),
    Example("I had him for Automata and he was a great professor. He is very knowledgeable and his lectures are very clear. He is very strict on the homeworks and exams, but he curves the exams. He is very nice and helpful. I would recommend him.", "positive"),
    Example("I took Pervin for Automata and I would recommend him. When I took it he gave very frequent, short, and easy homework that helps retain the material. His HW has no weights in final grade, that can be a good or bad thing depending on you. His exams are basically aggregated homeworks so they're not that difficult.I found his lectures to be quite clear and helpful with understanding the materials. If you just go to his lectures and do all the homework early I think Pervin's class will be pretty good.Oh and he grades stuff lightning fast too. I'm talking 10 minutes after submitting a hw.", "positive"),
    Example("Gupta is just a really rude and unapproachable guy who seems to be a professor to just feel a superior to undergrads. He has no interest in teaching you anything and has more of an interest in making you feel dumb.He's really really bad at teaching, incredibly disorganized, reads off really ugly slides and has no clear lesson plan.Neeraj Gupta represents not just the bottom tier for Professors at UTD but the bottom tier for humanity in general.", "negative"),
    Example("I had Gupta for Computer Networks instead of Automata, but I did not like his class at all. He is knowledgeable but very disorganized. He uploads vague assignments typed straight into the box on eLearning filled with typos. Not funIf he's teaching it next semester, I highly recommend Stallbohm for Automata. Stallbohm is an excellent professor who goes out of his way to prepare detailed, helpful notes on the topics and is always responsive to student questions", "negative"),
    Example("He curves heavily, I can’t speak for her tests but really if you just do the homework you would be fine for his class. He’s just a good overall professor tho, he will actually teach so you will learn the material", "positive"),
    Example("Jason Smith is really hard from what I've heard, but you will learn a lot from his class. I've heard that Dollinger isn't the best but I don't know for sure since I haven't taken him.", "positive"),
    Example("Do not take Jason Smith if you are working part-time or have a rigorous course load. If you aren't willing to spend a copious amount of time on his assignments, you will most likely fail his class. Of course, look at UTD Grades, RMP, and coursebook before gauging a specific professor.", "negative"),
    Example("I took Smith for 2337 and I will say that his UTD Grades and RMP are very misleading. He is easily the best professor I've had at this school and you should 100% take him if you want to actually learn programming. The coursework is a bit high but you're getting actual practice in not useless fluff. It's completely doable if you're actually willing to put in the work and start on the projects early. His lectures are also great and he actually cares about his students. Take Professor Smith. Future you will be thankful.", "positive"),
    Example("I took Chida and I would recommend her strongly. She makes complex stuff like Dynamic Prog and Graph Algos so much easier, just her way of teaching and her examples are super easy to follow. Her exams are also like her examples, however all 3 of her exams have a question where you need to write a pseudocode for a problem on the spot - that's the only part I found tricky.", "positive"),
    Example("I took chitturi. He's decent. He has a thick accent and is hard to understand at times. He is available after class which is cool. His grades are mainly based on exams... decent overall though.", "positive"),
]


def professorSearch(professorName):
    profComments = []
    profRating = 0.0
    for submission in subreddit.search(professorName, limit=10):
        for comment in submission.comments:
            if(len(comment.body) <= 512):
                profComments.append(comment.body)
    if(len(profComments) == 0):
        return 0.0
    response = co.classify(
        model='medium', inputs=profComments, examples=cohereExamples)
    for item in response.classifications:
        if item.prediction == "positive":
            profRating += item.confidence
        elif item.prediction == "negative":
            profRating -= item.confidence
    return profRating/len(profComments)


if __name__ == "__main__":
    args = sys.argv[1:]
    print(professorSearch(args[0]))
