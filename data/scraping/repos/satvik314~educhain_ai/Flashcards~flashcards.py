import streamlit as st
import json
import random
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

import os
os.environ["OPENAI_API_KEY"] = st.secrets["API_KEY"]

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
    
difficulty = [1,2,4,7,15,25]
def main():
    st.title("Flashcards")
    
    
    #declaring all the sessions state variables required
    if 'data' not in st.session_state:
        st.session_state.data = {}
    if 'day_started' not in st.session_state:
        st.session_state.day_started = False  
    if 'pre_process' not in st.session_state:
        st.session_state.pre_process = False
    if 'display_topic' not in st.session_state:
        st.session_state.display_topic = False
    if 'start_flashcard' not in st.session_state:
        st.session_state.start_flashcard = False
    if 'template' not in st.session_state:
        st.session_state.template = ''
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'topic' not in st.session_state:
        st.session_state.topic = ''
    if 'topic_no' not in st.session_state:
        st.session_state.topic_no = 0
    if 'valid' not in st.session_state:
        st.session_state.valid = False
    if 'difficulty' not in st.session_state:
        st.session_state.difficulty = []
    if 'generate' not in st.session_state:
        st.session_state.generate = False
    if 'generated' not in st.session_state:
        st.session_state.generated = False
    if 'card_index' not in st.session_state:
        st.session_state.card_index = 0
    if 'flashcard' not in st.session_state:
        st.session_state.flashcard = ''
    if 'flashcards' not in st.session_state:
        st.session_state.flashcards = []
    if 'remembered' not in st.session_state:
        st.session_state.remembered = []
    if 'total_cards' not in st.session_state:
        st.session_state.total_cards = 5
    if 'show_cards' not in st.session_state:
        st.session_state.show_cards = False
    if 'stats' not in st.session_state:
        st.session_state.stats = False
    if 'lesson' not in st.session_state:
        st.session_state.lesson = ''
    if 'daily' not in st.session_state:
        st.session_state.daily = True
    if 'select_topiic' not in st.session_state:
        st.session_state.select_topiic = False
        
    #All the functions used are declared below
    
    # Called when a topic in the lesson is completed to reset all the associated session state variables
    def topic_complete():
        st.session_state.valid = False
        st.session_state.generate = False
        st.session_state.generated = False
        st.session_state.template = ''
        st.session_state.topic = ''
        st.session_state.topic_no += 1
        st.session_state.card_index = 0
        st.session_state.flashcard = ''
        st.session_state.flashcards = []
        st.session_state.remembered = []
        st.session_state.total_cards = 5
        st.session_state.start_flashcard = True
    
    # This is called when all the topics of a lesson of a day is completed. This updates the difficulty and related variables in the json file.
    def day_complete():
        st.session_state.pre_process = False
        st.session_state.topics = []
        st.session_state.difficulty = []
        st.session_state.data = {}
        st.session_state.topic_no = 0
        st.session_state.stats = False
        st.session_state.display_topic = False
        st.session_state.lesson = ''
        st.session_state.start_flashcard = False

    # Next four functions defines functionalities of remembering the card and jumping to the next card
    def Next():
            st.session_state.flashcard = st.session_state.flashcards[st.session_state.card_index]
    def yes():
        st.session_state.remembered.append(1)
        st.session_state.card_index += 1
        if st.session_state.card_index < st.session_state.total_cards:
            Next()
    def no():
        st.session_state.show_cards = False
        st.session_state.remembered.append(0)
    def nextno():
        st.session_state.show_cards = True
        st.session_state.card_index += 1
        if st.session_state.card_index < st.session_state.total_cards:
            Next()
    
    # show stats
    def statistics():
        st.session_state.day_started = False
        st.session_state.stats = True
    
    # start showing the flashcards, i.e., the lesson assigned is starting    
    def daily_lesson():
        st.session_state.display_topic = False
        st.session_state.start_flashcard = True
    
    # if recommended topic is not chosen, custom choose the topic
    def select_topic():
        st.session_state.select_topiic = True
    
    # topic is selected, stop showing the interface for showing the topic
    def topic_selected():
        st.session_state.select_topiic = False
        daily_lesson()
    
    # pre-processing the json file every day. The counter and days_since_studying is decreased for the whole file. Then the topic is selected.    
    if st.session_state.pre_process == False: 
        with open("topic.json", 'r') as json_file:
            data = json.load(json_file)
            st.session_state.data = data
    
        for board in data["boards"]:
            for class_info in board["classes"]:
                for subject_info in class_info["subjects"]:
                    for lesson_info in subject_info["lessons"]:
                        lesson_info["counter"] -= 1
                        lesson_info["days_since_studying"] += 1
                        if lesson_info["counter"] <= 0:
                            for topic_info in lesson_info["topics"]:
                                st.session_state.topics.append(topic_info["name"])
                            st.session_state.lesson = lesson_info["name"]
                            break
                    if st.session_state.lesson == '':
                        lesson = subject_info["lessons"][random.randint(0, len(subject_info["lessons"]) - 1)]
                        for topic_info in lesson["topics"]:
                            st.session_state.topics.append(topic_info["name"])
                        st.session_state.lesson = lesson["name"]
        st.session_state.pre_process = True
        st.session_state.day_started = True
        st.session_state.display_topic = True
    
    # Custom select the topics from the drop-down list            
    if st.session_state.select_topiic:
        st.session_state.day_started = True
        st.session_state.display_topic = False
        st.session_state.topics = []
        st.session_state.difficulty = []
        st.session_state.topic_no = 0
        st.session_state.stats = False
        st.session_state.lesson = ''
        data = st.session_state.data        
        
        board_placeholder=st.empty()
        class_placeholder=st.empty()
        subject_placeholder = st.empty()
        lesson_placeholder = st.empty()
        topic_placeholder = st.empty()
        button=st.empty()
        
        board_names = [board["name"] for board in data["boards"]]
        board = board_placeholder.selectbox("Select board",board_names)
        board_list = next((b for b in data["boards"] if b["name"] == board), None)

        class_names = [Class["name"] for Class in board_list["classes"]]
        classe = class_placeholder.selectbox("Select class",class_names)
        classe_list= next((b for b in board_list["classes"] if b["name"]==classe) , None)

        subject_names = [subject["name"] for subject in classe_list["subjects"]]
        subject = subject_placeholder.selectbox("Select subject",subject_names)
        subject_list= next((b for b in classe_list["subjects"] if b["name"]==subject) , None)

        lesson_names = [lesson["name"] for lesson in subject_list["lessons"]]
        lesson = lesson_placeholder.selectbox("Select Lesson",lesson_names)
        lesson_list= next((b for b in subject_list["lessons"] if b["name"]==lesson) , None)
        
        topic_names = [topic["name"] for topic in lesson_list["topics"]]
        
        st.session_state.lesson = lesson
        st.session_state.topics = []
        topic = topic_placeholder.selectbox("Select topic", topic_names)
        st.session_state.topics.append(topic)
        
        if button.button("Generate", on_click=topic_selected):
            if board and classe and subject and lesson and topic :
                try:
                    board_placeholder.empty()
                    class_placeholder.empty()
                    subject_placeholder.empty()
                    lesson_placeholder.empty()
                    topic_placeholder.empty()
                except Exception as e :
                    st.error("Please select valid topic and number")
            
    
    # Daily lesson is recommended. From here, the user can continue with suggestion or custom select the topic.           
    if st.session_state.pre_process and len(st.session_state.topics)>0 and st.session_state.topic_no < len(st.session_state.topics) and st.session_state.display_topic:
        
        st.subheader(f"The lesson recommended for today is {st.session_state.lesson}")
        st.button("Go on with the lesson", on_click=daily_lesson)
        st.button("Custom select topics", on_click=select_topic)
    
    # validating that the pre-processing is done and topics for flashcards are available    
    elif st.session_state.pre_process and len(st.session_state.topics)>0 and st.session_state.topic_no < len(st.session_state.topics) and st.session_state.start_flashcard:
        st.session_state.valid = True
    
    # Flashcards are completed. Store the collected data from the user back into the json file.    
    elif st.session_state.day_started and st.session_state.pre_process and (len(st.session_state.topics)==0 or st.session_state.topic_no >= len(st.session_state.topics)):
        st.write("Completed")
        j = 0
        for board in st.session_state.data["boards"]:
            for class_ in board["classes"]:
                for subject in class_["subjects"]:
                    for lesson in subject["lessons"]:
                        if lesson["name"] == st.session_state.lesson:
                            x = 0
                            y = 0
                            for topic in lesson["topics"]:
                                if topic["name"] in st.session_state.topics:
                                    topic["revised"] += 1
                                    index = st.session_state.topics.index(topic['name'])
                                    if st.session_state.difficulty[index] == 1:
                                        if topic['difficulty'] in difficulty:
                                            curr_diff = difficulty.index(topic['difficulty'])
                                            if curr_diff != difficulty[-1]:
                                                topic['difficulty'] = difficulty[difficulty.index(topic['difficulty'])]
                                    elif st.session_state.difficulty[index] > 0.5:
                                        topic['difficulty'] = difficulty[1]
                                    else:
                                        topic['difficulty'] = difficulty[0]
                                    x += 1
                                    y += topic["difficulty"]/difficulty[2]
                            lesson["difficulty"] = int(y)
                            lesson["revised"] += x
                            lesson["counter"] = lesson["counter"] + abs((int(y))+0.5*(lesson["days_since_studying"] - x/len(lesson["topics"])))
                            
        st.session_state.start_flashcard = False    
        with open("topic.json", 'w') as file:
            json.dump(st.session_state.data, file, indent=2)
        st.button("View Statistics", on_click=statistics)
        st.button("Click here to revise more", on_click=select_topic)
        st.button("End Revision", on_click=day_complete)
    
    # view stats. both lesson wise and topic wise
    if st.session_state.stats:
        data = st.session_state.data
        boards = data["boards"]
        selected_board = st.selectbox("Select Board", [board["name"] for board in boards])

        for board in data["boards"]:
            if board["name"] == selected_board:
                classes = board["classes"]
                selected_class = st.selectbox("Select Class", [class_["name"] for class_ in classes])

                for class_ in classes:
                    if class_["name"] == selected_class:
                        subjects = class_["subjects"]
                        selected_subject = st.selectbox("Select Subject", [subject["name"] for subject in subjects])

                        for subject in subjects:
                            if subject["name"] == selected_subject:
                                lessons = subject["lessons"]
                                bar_chart_data = {lesson["name"]: 100/lesson["difficulty"] for lesson in lessons}
                                st.subheader("Lessons Difficulty")
                                st.bar_chart(bar_chart_data)
                                
                                selected_lesson = st.selectbox("Select Lesson to see topic-wise", [lesson["name"] for lesson in lessons])
                                
                                easy_lesson = subject['lessons'][0]['name']
                                hard_lesson = easy_lesson
                                for lesson in lessons:
                                    if lesson["name"] == selected_lesson:
                                        topics = lesson["topics"]
                                        bar_chart_data1 = {topic["name"]: 100/topic["difficulty"] for topic in topics}
                                        st.subheader("Topic-wise Distribution")
                                        st.bar_chart(bar_chart_data1)
        st.write()
        st.button("End Revision", on_click=day_complete)
        st.button("Click here to revise more", on_click=select_topic)
    
    # Generatint the template for flashcards    
    if st.session_state.generate == False and st.session_state.valid:
        topic = st.session_state.topics[st.session_state.topic_no]
        st.session_state.topic = topic
        content_template = "Give me 5 summary points on {topic}.Follow format: start new point from next line. Don't leave a line in between "
        template = content_template
        st.session_state.template = template
        st.session_state.generate = True
    
    # Changing the subheader to topic.
    if st.session_state.generate:
        st.subheader(st.session_state.topic)
    
    # Generate the flashcards using the GPT prompt. 
    if st.session_state.generate and st.session_state.generated == False:
        topic = st.session_state.topic
        content_prompt = PromptTemplate(template=st.session_state.template, input_variables=['topic'])
        gpt3_model = ChatOpenAI(temperature=0.5)
        discourse_writer = LLMChain(prompt=content_prompt, llm=gpt3_model)
        posts = discourse_writer.run(topic=topic)
        lines = posts.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip() != '']
        if len(lines) != 0:
            
            st.session_state.flashcards = lines
            st.session_state.flashcard = lines[0]
            st.session_state.generated = True
            st.session_state.show_cards = True        

    # displaying the flashcards
    if st.session_state.generated and st.session_state.card_index < st.session_state.total_cards and st.session_state.show_cards:
        
        st.write(f"<div class='flashcard cards'> {st.session_state.flashcard}</div>", unsafe_allow_html=True)
        st.write("Remebered it?")
        st.button("Yes", on_click=yes)
        st.button("No", on_click=no)
        st.write(f"Flashcard {st.session_state.card_index + 1} of {st.session_state.total_cards}")

    # if the card is not remembered, explain the point in detail
    if st.session_state.generated and st.session_state.card_index < st.session_state.total_cards and st.session_state.show_cards == False:
        topic = st.session_state.flashcard
        template = "Explain in around 70-90 words the point \"{topic}\" "
        st.session_state.template = template
        content_prompt = PromptTemplate(template=st.session_state.template, input_variables=['topic'])
        gpt3_model = ChatOpenAI(temperature=0.5)
        discourse_writer = LLMChain(prompt=content_prompt, llm=gpt3_model)
        posts = discourse_writer.run(topic=topic)
        st.write(f"<div class='flashcard cards'> {topic}</div>", unsafe_allow_html=True)
        st.write("\n\n")
        st.write(f'<div class="flashcard cards"> {posts} </div> ', unsafe_allow_html=True)
        st.button("Next", on_click=nextno)
     
    # At the end of each topic, display how many cards the student remembered       
    if st.session_state.generated and st.session_state.card_index >= st.session_state.total_cards:
        st.write(f"Flashcards completed for the topic {st.session_state.topic}")
        i = 0
        for i in range(st.session_state.total_cards):
            st.write(f"<div class='flashcard cards r{st.session_state.remembered[i]}'> {st.session_state.flashcards[i]}</div>", unsafe_allow_html=True)
        res = f"You remembered {sum(st.session_state.remembered)} out of {st.session_state.total_cards} cards"
        st.session_state.difficulty.append(sum(st.session_state.remembered)/st.session_state.total_cards)
        if sum(st.session_state.remembered) == st.session_state.total_cards:
            st.write(f"<div class='flashcard congratulation'> Congratulations </div>", unsafe_allow_html=True)
        elif sum(st.session_state.remembered) - st.session_state.total_cards >= -2:
            st.write(f"<div class='flashcard keep_it_up'> Keep it up </div>", unsafe_allow_html=True)
        else:
            st.write(f"<div class='flashcard revise'> Good attempt </div>", unsafe_allow_html=True)
        st.write(res)
        st.button("Continue", on_click=topic_complete)   
    
if __name__ == '__main__':
    main()
