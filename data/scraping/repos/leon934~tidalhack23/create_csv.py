from openai_curriculum_creator import interest_form, cherry_questions
import pandas as pd

interest_questions = {
    "1. In tennis, what does it mean when the derivative of a player's position with respect to time is equal to zero?": {
        "A": "The player is not moving",
        "B": "The player is moving at a constant speed",
        "C": "The player is hitting the ball",
        "D": "The player is about to serve",
        "Answer": "B"
    },

    "2. During a tennis match, if a player's velocity graph shows a horizontal line, what can we conclude about the player's acceleration?": {
        "A": "Acceleration is positive",
        "B": "Acceleration is negative",
        "C": "Acceleration is zero",
        "D": "Acceleration is constant",
        "Answer": "C"
    },

    "3. If the derivative of a player's position function is zero at a specific moment in a tennis game, what does this imply about the player's motion?": {
        "A": "The player is moving in the positive direction",
        "B": "The player is moving in the negative direction",
        "C": "The player has changed direction",
        "D": "The player is accelerating",
        "Answer": "C"
    },

    "4. How does the concept of the derivative being zero relate to a tennis player at the peak of their jump during a serve?": {
        "A": "The player's speed is zero",
        "B": "The player's acceleration is zero",
        "C": "The player is about to hit the ball",
        "D": "The player is about to land",
        "Answer": "B"
    },

    "5. In tennis, if a player's position function is given by a constant, what can we say about the derivative of the position function?": {
        "A": "The derivative is zero",
        "B": "The derivative is positive",
        "C": "The derivative is negative",
        "D": "The derivative is undefined",
        "Answer": "A"
    }
}

questions = {
    "1. What does the first derivative test help determine?": {
        "A": "Concavity of a function",
        "B": "Location of critical points",
        "C": "Continuity of a function",
        "D": "Limit of a function",
        "Answer": "B"
    },

    "2. In the first derivative test, a critical point where the derivative changes sign indicates that the function has a __________ at that point.": {
        "A": "Local minimum",
        "B": "Local maximum",
        "C": "Point of inflection",
        "D": "Vertical asymptote",
        "Answer": "C"
    },

    "3. If the first derivative is positive to the left of a critical point and negative to the right, what type of extremum does the function have at that point?": {
        "A": "Local minimum",
        "B": "Local maximum",
        "C": "Point of inflection",
        "D": "Neither minimum nor maximum",
        "Answer": "A"
    },

    "4. How is the critical point related to the first derivative in the first derivative test?": {
        "A": "The first derivative is zero at the critical point.",
        "B": "The first derivative is positive at the critical point.",
        "C": "The first derivative is negative at the critical point.",
        "D": "The first derivative is undefined at the critical point.",
        "Answer": "A"
    },

    "5. The first derivative test is based on analyzing changes in the sign of the first derivative around critical points to determine the __________ of the function.": {
        "A": "Continuity",
        "B": "Concavity",
        "C": "Monotonicity",
        "D": "Discontinuity",
        "Answer": "C"
    }
}

def student_question(questions) -> list:
    answers = []
    
    student_name = input("What is your name? ")
    answers.append(student_name)

    for key in questions.keys():
        print(key)
        
        for choice in questions[key].keys():
            if choice != "Answer":
                print(choice + ": " + questions[key][choice])

        input_answer = input("What is your answer? ").upper()
        
        if input_answer == questions[key]["Answer"]:
            answers.append(1)
        else:
            answers.append(0)
    return answers

# Serves to engage the students to increase their engagement and productivity.
def interest_learning():
    # Caters towards the interests of the parameter demographic.
    desired_demographic = input("What demographic do you want to cater towards? (college students, middle school students, etc.)")

    interest_curriculum = interest_form(desired_demographic)

    print(interest_curriculum[0])

    for key in interest_curriculum[1].keys():
        print(key)
        
        for choice in interest_curriculum[1][key].keys():
            if choice != "Answer":
                print(choice + ": " + interest_curriculum[1][key][choice])

        input_answer = input("What is your answer? ").upper()
        
        if input_answer == interest_curriculum[1][key]["Answer"]:
            print("That's correct!")
        else:
            print("That's incorrect. The correct answer is " + interest_curriculum[1][key]["Answer"] + ".")

def main():
    question = []
    columns_to_add = {}
    answer_again, interest_learning = "Y", 'Y'

    # Cherry questions: generates questions about the topic.
    desired_topic = input("What topic do you want to generate questions about?")

    # Optional interest learning: generates a curriculum based on the user's interests.
    interest_learning_input = input("Do you want to generate a curriculum based on your topic that involves your interests? (Y/N)")

    if interest_learning_input == "Y":
        interest_learning()

    # The variable question is a dictionary.
    questions = cherry_questions(desired_topic, 5)

    for question in questions.keys():
        columns_to_add[question] = 0

    topic_df = pd.DataFrame(columns_to_add, index=[0])

    while answer_again != "N":
        # Asks the student for their name and asks the questions.
        student_answers = student_question(question)

        # Creates a dataframe with the columns being the questions and the rows being the student answers with their name.
        topic_df.loc[len(topic_df.index)] = student_answers

        answer_again = input("Is there another student that wants to answer the questions? (Y/N)")

    # Saves the dataframe to a csv file.
    topic_df.to_csv('./STUDENT_ANSWERS.csv')

if __name__ == '__main__':
    main()