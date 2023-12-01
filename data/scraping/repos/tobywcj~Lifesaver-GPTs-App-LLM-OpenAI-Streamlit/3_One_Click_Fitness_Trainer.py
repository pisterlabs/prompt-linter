import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import(
    SystemMessage, # set the behavior of the assistant
    HumanMessage, # what we ask
    AIMessage #  store prior responses
)




def validate_openai_api_key(api_key):
    import openai

    openai.api_key = api_key

    with st.spinner('Validating API key...'):
        try:
            response = openai.Completion.create(
                engine="davinci",
                prompt="This is a test.",
                max_tokens=5
            )
            # print(response)
            validity = True
        except:
            validity = False

    return validity


def fitness_plan_prompt(llm, age, gender, height, current_weight, medical_conditions,
                       food_allergies, fitness_goals, workout_days, exercise_preference,
                       diet_preference, meals_per_day, snacks_per_day, foods_disliked,
                       language='English'):
    
    from langchain.chains import LLMChain
    from langchain import PromptTemplate

    template = ''' 
        Take the following information about me and create a custom diet and exercise plan. 
        I am {Age} years old, {Gender}, {Height} cm. your current weight is {Currentweight} kg. 
        your current medical conditions are {MedicalConditions}. 
        I have food allergies to {FoodAllergies}. 
        your primary fitness and health goals are {PrimaryFitnessHealthGoals}. 
        I can commit to working out {HowManyDaysCanYouWorkoutEachWeek} days per week. 
        I prefer and enjoy his type of workout {ExercisePreference}. 
        I have a diet preference {DietPreference}. 
        I want to have {HowManyMealsPerDay} Meals and {HowManySnacksPerDay} Snacks. 
        I dislike eating and cannot eat {ListFoodsYouDislike}. 

        Create a summary of your diet and exercise plan. 
        Create a detailed workout program for your exercise plan. 
        Create a detailed Meal Plan for your diet. 
        Create a detailed Grocery List for your diet that includes quantity of each item. Avoid any superfluous pre and post descriptive text. 
        Don't break character under any circumstance. 
        Include a list of 10 motivational quotes that will keep me inspired towards your goals.
        written in {language}.'''

    prompt = PromptTemplate(
        input_variables=["Age",
                        "Gender",
                        "Height",
                        "Currentweight",
                        "MedicalConditions",
                        "FoodAllergies",
                        "PrimaryFitnessHealthGoals",
                        "HowManyDaysCanYouWorkoutEachWeek",
                        "ExercisePreference",
                        "DietPreference",
                        "HowManyMealsPerDay",
                        "HowManySnacksPerDay",
                        "ListFoodsYouDislike",
                        "language"
                        ],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({"Age": age,
                        "Gender": gender,
                        "Height": height,
                        "Currentweight": current_weight,
                        "MedicalConditions": medical_conditions,
                        "FoodAllergies": food_allergies,
                        "PrimaryFitnessHealthGoals": fitness_goals,
                        "HowManyDaysCanYouWorkoutEachWeek": workout_days,
                        "ExercisePreference": exercise_preference,
                        "DietPreference": diet_preference,
                        "HowManyMealsPerDay": meals_per_day,
                        "HowManySnacksPerDay": snacks_per_day,
                        "ListFoodsYouDislike": foods_disliked,
                        "language": language
                        })

    return output


# clear the chat history from streamlit session state
def clear_history():
    if 'fitness_history' in st.session_state:
        del st.session_state.fitness_history
        st.session_state.fitness_history = [SystemMessage(content='You are a highly renowned health and nutrition Fitness expert.')]


if __name__ == "__main__":

    ############################################################ System Configuration ############################################################

    # creating the messages (chat history) in the Streamlit session state
    if 'fitness_history' not in st.session_state:
        st.session_state.fitness_history = [SystemMessage(content='You are a highly renowned health and nutrition Fitness expert.')]

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

    ############################################################ SIDEBAR widgets ############################################################

    with st.sidebar:

        # Setting up the OpenAI API key via secrets manager
        if 'OPENAI_API_KEY' in st.secrets:
            api_key_validity = validate_openai_api_key(st.secrets['OPENAI_API_KEY'])
            if api_key_validity:
                os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
                st.success("✅ API key is valid and set via Encrytion provided by Streamlit")
            else:
                st.error('🚨 API key is invalid and please input again')
        # Setting up the OpenAI API key via user input
        else:
            api_key_input = st.text_input("OpenAI API Key", type="password")
            api_key_validity = validate_openai_api_key(api_key_input)

            if api_key_input and api_key_validity:
                os.environ['OPENAI_API_KEY'] = api_key_input
                st.success("✅ API key is valid and set")
            elif api_key_input and api_key_validity == False:
                st.error('🚨 API key is invalid and please input again')

            if not api_key_input:
                st.warning('Please input your OpenAI API Key')
        
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

        st.divider()

        # Required Customer Information
        st.subheader('Customer Information')

        age = st.number_input('Age', min_value=16, max_value=150, value=30, step=1)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170, step=1)
        current_weight = st.number_input('Current Weight (kg)', min_value=10, max_value=300, value=100, step=1)
        medical_conditions = st.text_input('Medical Conditions')
        food_allergies = st.text_input('Food Allergies')
        fitness_goals = st.selectbox('Primary Fitness Health Goals', ["Weight loss", "Muscle gain", "Cardiovascular fitness", "Strength and endurance", "Flexibility and mobility", "Overall health and well-being", "Body composition improvement", "Stress reduction", "Injury prevention and rehabilitation", "Performance enhancement"])
        workout_days = st.number_input('How many days can you workout each week?', min_value=1, max_value=7, value=3, step=1)
        exercise_preference = st.selectbox('Exercise Preference', ['Cardio', 'Strength', 'Flexibility'])
        diet_preference = st.selectbox('Diet Preference', ['High Protein', 'Low Carb', 'Low Fat'])
        meals_per_day = st.number_input('How many meals per day?', min_value=1, max_value=10, value=3, step=1)
        snacks_per_day = st.number_input('How many snacks per day?', min_value=1, max_value=10, value=3, step=1)
        foods_disliked = st.text_input('List Foods You Dislike')
        language = st.text_input('Language')

        if st.button('Create your Personal Fitness Plan'):
            if api_key_validity:
                if age and gender and height and current_weight and medical_conditions and food_allergies and fitness_goals and workout_days and exercise_preference and diet_preference and meals_per_day and snacks_per_day and foods_disliked:
                    with st.spinner('Creating your personal fitness plan ...'):
                        personal_plan = fitness_plan_prompt(llm, age, gender, height, current_weight, medical_conditions,
                                food_allergies, fitness_goals, workout_days, exercise_preference,
                                diet_preference, meals_per_day, snacks_per_day, foods_disliked,
                                language)
                    
                    # adding the response's content to the session state
                    st.session_state.fitness_history.append(AIMessage(content=personal_plan))
                elif not age or not gender or not height or not current_weight or not medical_conditions or not food_allergies or not fitness_goals or not workout_days or not exercise_preference or not diet_preference or not meals_per_day or not snacks_per_day or not foods_disliked:
                    st.warning('Please fill in all the fields.')
            elif not api_key_validity:
                st.warning('Please enter your OpenAI API key to continue.')

        if st.button('Clear Chat History'):
            clear_history()


    ############################################################ MAIN PAGE widgets ############################################################

    st.title('🏋🏽 One Click Fitness Trainer')

    st.divider()

    if len(st.session_state.fitness_history) :
        st.chat_message('assistant').markdown('''Welcome to One-Click-Fitness-Trainer Centre! 
                                              \nSpend 30 secs to create your personal fitness plan, including the following:
                                              \n- a summary of your daily diet and exercise plan
                                              \n- a detailed weekly workout program for your exercise plan
                                              \n- a detailed Meal Plan for your diet
                                              \n- a detailed Grocery List for your diet that includes quantity of each item 
                                              \n- a list of 10 motivational quotes
                                              \nAsk anything on your customized plan, simply fill in the blanks in the side bar and CLICK.''')

    # if the user entered a question
    question = st.chat_input(placeholder="Ask me anything about Fitness and Health")

    if question:
        if api_key_validity:
            st.session_state.fitness_history.append(
                HumanMessage(content=question)
            )

            with st.spinner('Working on your request ...'):
                # creating the ChatGPT response
                response = llm(st.session_state.fitness_history)

            st.session_state.fitness_history.append(AIMessage(content=response.content))
        elif not api_key_validity:
            st.warning('Please enter your OpenAI API key to continue.')

    # displaying the messages (chat history)
    for message in st.session_state.fitness_history[1:]:
        if isinstance(message, HumanMessage):
            st.chat_message('user').markdown(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message('assistant').markdown(message.content)
