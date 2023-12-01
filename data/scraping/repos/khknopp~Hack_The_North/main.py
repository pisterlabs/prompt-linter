from taipy.gui import Gui, Html, navigate, notify, State
from Pages.about import about_md
from Pages.home import home_md
from Pages.watching import watching_md
from Pages.video import createMarkdown
from Pages.landingp import landingp_md
from Pages.landingpt import landingpt_md
from Pages.landingcs import landingcs_md
from Pages.quiz import quiz_md
import time
from video_utils import *
from question_generator import *
from summarize import *
from dotenv import load_dotenv
import os
import cohere
import psycopg
from psycopg.errors import SerializationFailure, Error
from psycopg.rows import namedtuple_row
import pickle as pk
from db import get_params
import threading
from gaze_tracking import GazeTracking
import cv2


load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
COCKROACH_USERNAME = os.getenv('COCKROACH_USERNAME')
COCKROACH_PASSWORD = os.getenv('COCKROACH_PASSWORD')

db_url = f"postgresql://{COCKROACH_USERNAME}:{COCKROACH_PASSWORD}@cuter-falcon-5491.g8z.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"

conn = psycopg.connect(db_url, application_name="$ defaultdb", row_factory=namedtuple_row)

with conn.cursor() as cur:
    cur.execute(
        "DROP table videos"
    )
conn.commit()

video = get_params("p2J7wSuFRl8", "Lecture 1 - Philosophy of Death")
video.add_video(conn)
video.update_db(conn, "p2J7wSuFRl8")

# co = cohere.Client(COHERE_API_KEY)
# db_url = f"postgresql://{COCKROACH_USERNAME}:{COCKROACH_PASSWORD}@cuter-falcon-5491.g8z.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"


# conn = psycopg.connect(db_url, application_name="$ defaultdb", row_factory=namedtuple_row)

generalNavigation = [("/home", "Home"), ("/about", "About"), ("/watching", "Watching")]
watchingNavigation = [("/watching/video", "Video"), ("/watching/history", "History")]

root_md = """
<center>
<|navbar|lov={generalNavigation}|>
</center>

"""

        
#add_video(conn, "9syvZr-9xwk", "This is a summary of the video", ["These are the closed questions"], ["These are the closed answers"], ["These are the open questions"], ["These are the open answers"], "This is the title", "This is the transcript")
video_md = createMarkdown("https://www.youtube.com/embed/p2J7wSuFRl8")
pages = {
    "/": root_md,
    "home": home_md,
    "about": about_md,
    "watching": watching_md,
    "watching/quiz": quiz_md,
    "watching/video": video_md,
    "watching/landingp": landingp_md,
    "watching/landingpt": landingpt_md,
    "watching/landingcs": landingcs_md
}

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

text = ""

initialOpen = time.time()

isLookingAway = False

lookedAwayStart = 0
lookedAwayEnd = 0
urlLink = ""
allIntervals = []


def camera():
    global allIntervals
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        new_frame = gaze.annotated_frame()
        hor = gaze.horizontal_ratio()
        ver = gaze.vertical_ratio()

        if(hor != None and ver != None):
            if(hor > 0.9 or ver > 0.9 or ver < 0.1 or hor < 0.1):
                text = "Not paying attention!"
                lookedAway()
            else:
                text = "Paying attention!"
                lookedBack()
        else:
            text = "Not paying attention!"
            lookedAway()

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)


        cv2.imshow("Demo", frame)

        if cv2.waitKey(1) == 27:
            break
    
    print(allIntervals)

    webcam.release()
    cv2.destroyAllWindows()

def lookedAway():
    global lookedAwayStart
    global isLookingAway
    global initialOpen
    if(isLookingAway == False):
        lookedAwayStart = time.time() - initialOpen
    isLookingAway = True
        
def lookedBack():
    global isLookingAway
    global lookedAwayStart
    global lookedAwayEnd
    global allIntervals
    global initialOpen
    if isLookingAway == True:
        lookedAwayEnd = time.time() - initialOpen
        if lookedAwayEnd - lookedAwayStart >= 3:
            allIntervals.append([lookedAwayStart, lookedAwayEnd])
        isLookingAway = False


if __name__ == "__main__":    
    b = threading.Thread(name='background', target=camera)
    b.start()


    questionNum = 0
    essayAnswer = ""

    lastNotes = """
    What is the purpose of a database index, and how does it improve query performance?
What is object-oriented programming, and how does it differ from procedural programming?

3:16
Introduction to Computer Science:
Computer Science 101 is a foundational course that introduces students to the fundamental concepts and principles of computer science.
Algorithmic Thinking:
Students learn how to think algorithmically, breaking down problems into step-by-step instructions that a computer can execute.
Programming Fundamentals:
The course covers programming basics, including variables, data types, control structures (such as loops and conditionals), and functions.
Data Structures:
Students are introduced to fundamental data structures like arrays, lists, stacks, and queues, understanding when and how to use them.
Basic Algorithms:
Essential algorithms, such as searching and sorting algorithms, are explored, along with their time and space complexity analysis.
Problem Solving:
Computer Science 101 emphasizes problem-solving skills, helping students approach complex issues logically and systematically.
Computer Architecture:
Basic computer architecture concepts are introduced, including the CPU, memory, and storage devices.
Software Development:
Students gain insights into the software development process, including design, coding, testing, and debugging.
Web Development (optional):
Some courses may include an introduction to web development, covering HTML, CSS, and JavaScript.
Ethical Considerations:
Ethical and social implications of computer science, including privacy and security, are discussed.
Real-World Applications:
Students explore real-world applications of computer science, from creating simple programs to understanding how technology impacts various industries.
Mathematical Concepts:
Computer Science 101 often includes mathematical concepts, such as logic, set theory, and basic discrete mathematics, which are essential for algorithm design.
Programming Languages:
Exposure to programming languages like Python, Java, or C++ is common, but the emphasis is on understanding core concepts rather than mastering a particular language.
Hands-On Projects:
The course typically involves hands-on coding projects and assignments to apply theoretical knowledge to practical problems.
Preparation for Further Study:
Computer Science 101 serves as a foundation for more advanced computer science courses, providing students with the skills and knowledge needed to pursue a deeper understanding of the field.
Computer Science 101 serves as a crucial starting point for anyone interested in computer science, regardless of their background or future career goals. It provides the necessary groundwork for more specialized and advanced studies in the field.
    """


    data = {
        "Number": range(1,5),
        "Question": ["What is the difference between a compiler and an interpreter in programming languages?", "Explain the Big O concept?", "What is the purpose of a database index, and how does it improve query performance?", "What is object-oriented programming, and how does it differ from procedural programming?"]
    }
    def on_menu(state, var_name, function_name, info):
        page = info['args'][0]
        navigate(state, to=page)

    def debug(state):
        print(state.allIntervals)

    Gui(pages=pages).run(title='NewApp', dark_mode=False)


# video = "9syvZr-9xwk"
#video = "4XGGPfaTcSo"
#video = "9syvZr-9xwk"
# video = "4XGGPfaTcSo"

#transcript, transcript_text = run_transcript(video)
# overall_summary = " ".join(summarize_text(co, transcript_text))
#separate based on fullstops to get bullet points

#open, closed, summary = split_execution(co, transcript_text)
#

#fragments = create_fragments(transcript, [[50, 100], [200, 230]])
#fragments = create_fragments(transcript, [[50, 100], [200, 230], [200, 230]])

#print(update_summary(co, summary, fragments))


#print(all_outputs)
# conn.close()