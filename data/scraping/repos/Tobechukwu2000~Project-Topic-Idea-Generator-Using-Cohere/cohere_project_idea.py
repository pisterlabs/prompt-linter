import streamlit as st
import cohere
import os

# Cohere API key
api_key = os.environ["COHERE_KEY"]

# Set up Cohere client
co = cohere.Client(api_key)


def generate_topic(department, favorite_topics):
     base_prompt_topic = textwrap.dedent("""
       Department: Mechanical Engineering
       Favorite Courses: Thermodynamics, Heat and mass transfer
       Project Topic: Refrigeration Using waste heat in cars.
       
       --SEPARATOR--
       
       Department: Mechanical Engineering
       Favorite Courses: Heat and mass transfer, Engineering drawing
       Project Topic: Analysis on effectiveness of a double heat exchanger with fin.
       
       --SEPARATOR--
       
       Department: Mechanical Engineering
       Favorite Courses: Metallurgy, Mechanics of fluid
       Project Topic: Aerodynamic modelling of wind Turbine Blades
       
       --SEPARATOR--
       
       Department: Civil Engineering
       Favorite Courses:  Strength of Materials, Civil Engineering Materials
       Project Topic: The effect of untreated sugarcane ash on the setting time and compressive strength of concrete mix
       
       --SEPARATOR--
       
       Department: Civil Engineering
       Favorite Courses: Concrete technology, Theory of structures
       Project Topic: Geometric Design of Highway
       
       --SEPARATOR--
       
       Department: Civil Engineering
       Favorite Courses: Soil mechanics, Environmental engineering
       Project Topic: The effect of Environment on bond resistance between concrete and steel reinforcement
       
       --SEPARATOR--
       
       Department: Electronics Engineering
       Favorite Courses: Circuit and systems, Physical and applied electronics
       Project Topic: UltraSonic Radar Project
       
       --SEPARATOR--
       
       Department: Electronics Engineering
       Favorite Courses: Electromagnetism
       Project Topic: Antenna Design
       
       --SEPARATOR--
       Department: Electrical Engineering
       Favorite Courses: Power Systems, Electrical Machines
       Project Topic: Solar wireless Electric Vehicle charging system
       
       --SEPARATOR--
       
       Department: Electrical Engineering
       Favorite Courses: Measurement and instrumentation
       Project Topic: E-bike speed controller systems
       
       --SEPARATOR--
       
       Department: Materials Engineering
       Favorite Courses: Engineering composites, Industrial metallurgy
       Project Topic: Glass Hybrid Fibres Epoxy Composite Material
       
       --SEPARATOR--
       
       Department: Materials Engineering
       Favorite Courses: Degradation of Metals and alloy, Advanced materials processing
       Project Topic: Investigating the degradation of Epoxy Resin Nanocomposite Exposed to different environment
       
       --SEPARATOR--
       
       Department: Agricultural and Bio-Resource Engineering
       Favorite Courses: Machinery and Food engineering
       Project Topic: Performance Evaluation of a solar powered poultry egg incubator
       
       --SEPARATOR--
       
       Department: Chemical Engineering
       Favorite Courses: Industry Chemistry
       Project Topic: Comparative study of Physicochemical analysis of borehole water in a municipal
       
       --SEPARATOR--
       
       Department: Chemical Engineering
       Favorite Courses: Petroleum Engineering
       Project Topic: Identification of well problems using well testing
       
       --SEPARATOR--
       
       Department: Computer Engineering
       Favorite Courses: Embedded systems, Circuit Analysis
       Project Topic: Water level controller using a microcontroller
       
       --SEPARATOR--
       
       Department: Mechanical Engineering
       Favorite Topics: Computer Aided Design(CAD), Control Systems, Automation
       Project Topic: Design and analysis of automated truck cabin suspension system
       
       --SEPARATOR--
       
       Department: Mechanical Engineering
       Favorite Topics: Computer Aided Design(CAD), Control Systems, Automation
       Project Topic: Detached-Eddy Simulations of Active Control systems on a simplified car geometry
       
       --SEPARATOR--
       
       Department: Mechanical Engineering
       Favorite Topics: Strength of materials, Solid mechanics, Internal Combustion Engine
       Project Topic: Transient Heat Conduction in a Solidifying Alloy
       
       --SEPARATOR--
       
       Department: Mechanical Engineering
       Favorite Topics: Strength of materials, Solid mechanics, Internal Combustion Engine
       Project Topic: Power Generation from Internal Combustion Engine using ATG 
       
       --SEPARATOR--
       Department:""")
    
    
    response = co.generate(
      model='xlarge',
      prompt= base_prompt_topic+" "+ department+"\nFavorite Topics: "+favorite_topics+"\nProject Topic: ",
      max_tokens=20,
      temperature=0.8,
      k=0,
      p=0.7,
      frequency_penalty=0,
      presence_penalty=0,
      stop_sequences=["--SEPARATOR--"],
      return_likelihoods='NONE')
    Project_topic = response.generations[0].text
    Project_topic = Project_topic.replace("\n\n--SEPARATOR--","").replace("\n--SEPARATOR--","").strip()
    return Project_topic


def generate_description(Project_topic):
     base_prompt_description = textwrap.dedent("""
       Project Topic: Mini Solar water heater
       Project description: This project involves the design and fabrication of a portable solar water heater
       
       --SEPARATOR--
       
       Project Topic: Bluetooth Gamepad for Android Gaming
       Project description: The gamepad will have a unique designed and shaped PCB in the shape of a gamepad. The PCB will have 2 x joysticks mounted on it for transmitting movement and aim commands to the phone
       
       --SEPARATOR--
       Project Topic: Aerodynamic Modelling of Wind turbine Blades
       Project description: The purpose of this project will be to find a simple linear modification to the shape of wind turbine blades
       
       --SEPARATOR--
       
       Project Topic: The effect of untreated sugarcane ash on the setting time and compressive strength of concrete mix
       Project description: The main objective of this work is to compare the compressive strength of concrete in which some percentages of cement had been replaced with equal weight of sugarcane ash with that of normal concrete produced from the same mix ratio, and to determine the effect of sugar cane ash on the initial and final setting time of concrete
       
       --SEPARATOR--
       
       Project Topic: Ultrasonic radar project
       Project description: Build a system that can monitor an area of limited range and alerts authorities with a buzzer as an alarm
       
       --SEPARATOR--
       
       Project Topic: Solar wireless electric vehicle charging system
       Project description: The system demonstrates how electric vehicles can be charged while moving on road, eliminating the need to stop for charging
       
       --SEPARATOR--
       
       Project Topic: Glass Hybrid fibres epoxy composite material
       Project description: This project involves the characterization of epoxy-based hybrid composites
       
       --SEPARATOR--
       
       Project Topic: Performance evaluation of a solar powered poultry egg incubator
       Project description: In this study, a solar photovoltaic powered chicken egg incubator was designed, fabricated and tested to evaluate its performance
       
       --SEPARATOR--
       
       Project Topic: Identification of well problems using well testing
       Project description: This project work is concerned with the use of well testing in identifying well problems
       
       --SEPARATOR--
       
       Project Topic:""")
        
    response = co.generate(
      model='xlarge',
      prompt=  base_prompt_description+" "+ Project_topic+"\nProject description:",
      max_tokens=50,
      temperature=0.8,
      k=0,
      p=0.7,
      frequency_penalty=0,
      presence_penalty=0,
      stop_sequences=["--SEPARATOR--"],
      return_likelihoods='NONE')
    Project_description = response.generations[0].text
    Project_description = Project_description.replace("\n\n--SEPARATOR--","").replace("\n--SEPARATOR--","").strip()
    return Project_description

st.title(" 💡 Project Idea Generator")
st.markdown('This app was built using the Generate model and endpoint provided by Cohere.' , unsafe_allow_html=False)

st.markdown(
    """<a href="https://www.cohere.ai/">Click here to visit their site and start building</a>""", unsafe_allow_html=True,
)
st.header('What is this app about?🤔')

st.markdown('The purpose of this app is to help engineering students generate creative topics for their projects based on their department. The fun part about this app is that it allows you to enter your favorite topics to be able to streamline project topic suggestions that will interest you. Cool right?😎 Let us dive right in!', unsafe_allow_html=False)

form=st.form(key="user_settings")
with form:
    department_input=st.text_input('DEPARTMENT(Enter the name of your department in full)', key='department_input')
    favorite_courses_input=st.text_input('FAVORITE COURSE(S)(If you have two or more favorite courses, please separate with commas)', key='favorite_courses_input')
    num_input=st.slider("Number of Project Topics", value=2, key='num_input', min_value=1, max_value=10)

    generate_button=form.form_submit_button('Generate Project Topic Ideas🚀')
    
    if generate_button:
        if department_input=="" or favorite_courses_input=="":
            st.error("Field cannot be blank")
            
        else:
            my_bar=st.progress(0.05)
            st.subheader('Project Ideas')
            for i in range(num_input):
                st.markdown("""---""")
                Project_topic=generate_topic(department_input, favorite_courses_input)
                Project_description= generate_description(Project_topic)
                st.markdown("#### "+ Project_topic)
                st.write(Project_description)
                my_bar.progress((i+1)/num_input)
                
st.header('About The creator👨')
st.markdown('Tobechukwu is a 400 level student of the Department of Mechanical Engineering, University of Nigeria Nsukka. He is a Machine Learning Engineer and an Artificial Intelligence enthusiast with keen interest in NLP and Computer Vision.', unsafe_allow_html=False)
