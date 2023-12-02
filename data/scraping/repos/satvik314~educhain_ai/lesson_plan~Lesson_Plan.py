import openai
import streamlit as st
from dotenv import load_dotenv
import os
from lesson_plan.Lesson_Utils import generate_response, template
from lesson_plan.students import student
import json

# setting model and api key:
load_dotenv()



def app():
    # Reading json file and storing its data in data variable:
    with open('lesson_plan/Syllabus1.json') as f:
        data = json.load(f)
        boards = []  # storing all available boards in json file to list
        for board in data['boards']:
            boards.append(board['name'])

    # creating student object
    stud = student()
    st.title("Lesson Plan Generator")
    # getting board as input from student with selectbox and setting attribute value:
    board = st.sidebar.selectbox('Select your board', boards)
    stud.set_board(board)

    # getting class data from student through selectbox:
    classes = []
    for board in data['boards']:
        if board['name'] == stud.get_board():
            for classs in board['classes']:
                classes.append(classs['name'])

    std = st.sidebar.selectbox('select your class', classes)  # class selected by student is stored here in std
    stud.set_std(std)  # added std attribute value

    # selecting subject to learn:
    subject_list = []
    for board in data['boards']:
        if board['name'] == stud.get_board():
            for classs in board['classes']:
                if classs['name'] == 'Class 7':
                    for subject in classs['subjects']:
                        subject_list.append(subject['name'])

    # showing list of available subjects in selectbox:
    subject = st.sidebar.selectbox("Select your subject", subject_list)
    stud.set_subject(subject)  # added subject attribute value

    # selecting lesson to learn:
    lesson_names = []
    for boards in data['boards']:
        if boards['name'] == stud.get_board():
            for classes in boards['classes']:
                if classes['name'] == stud.get_std():
                    for subject in classes['subjects']:
                        if subject['name'] == stud.get_subject():
                            for lesson in subject['lessons']:
                                lesson_names.append(lesson['name'])

    # showing list of lessons in subject selected by student and setting attribute value:
    lesson = st.sidebar.selectbox("Select lesson to learn", lesson_names)
    stud.set_lesson(lesson)

    # subtopics to learn:
    subtopics_names = ['All', ]
    for boards in data['boards']:
        if boards['name'] == stud.get_board():
            for classes in boards['classes']:
                if classes['name'] == stud.get_std():
                    for subject in classes['subjects']:
                        if subject['name'] == stud.get_subject():
                            for lesson in subject['lessons']:
                                if lesson['name'] == stud.get_lesson():
                                    subtopics_names.extend(lesson['topics'])
    # the above code will store all the subtopics of selected lesson in list object
    # creating selectbox to choose the subtopic:
    sub_topic = st.sidebar.selectbox("select subtopic:", subtopics_names)
    stud.set_subtopic(sub_topic)

    # selecting mode by default its learning mode:
    mode = st.sidebar.radio('Mode', ['Learning', 'Revision'])

    if mode == 'Learning':
        st.write("Mode: Learning")
    else:
        st.write("Mode: Revision")

    # button to generate response:
    if st.button("Generate"):
        # spinner will shown while generating response
        with st.spinner(f"Generating Lesson plan for {stud.get_std()} std of {stud.get_board()} student"):
            response = generate_response(template, mode, stud.get_std(), stud.get_lesson(), stud.get_board(),
                                         stud.get_subtopic())

        # Displaying the response:
        st.write(response)


# calling main function as code starts running:
# if __name__ == "__main__":
    # comprehension()
