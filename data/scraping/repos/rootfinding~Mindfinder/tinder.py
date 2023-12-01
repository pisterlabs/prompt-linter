import os
import json
import pandas as pd
import openai

with open('.creds') as f:
    creds = json.load(f)
    PINECONE_API_KEY = creds['PINECONE_API_KEY']
    PINECONE_ENVIRONMENT = creds['PINECONE_ENVIRONMENT']
    OPENAI_API_KEY = creds['OPENAI_API_KEY']

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
)
from src.agent_type import DialogueAgent, PineconeDialogueAgent
from src.simulation_type import DialogueSimulator

def select_next_speaker(step, agents):
    idx = step % len(agents)
    return idx

word_limit = 50  # word limit for task brainstorming


def generate_user(identity_agent, possible_match):
    quest = "This is a date to find the defects and positive values of the other person in that date and do actions to know each other and evaluate"

    identity_system_message = SystemMessage(
        content=(
            f"""{quest}
    Always remember that engaging in discussions involves both finding common ground and acknowledging differences with the other person.
    You are {identity_agent['name']}, and your date is {possible_match['name']}.
    Your character description is as follows: {identity_agent['description']}.
    During the conversation, you will have the opportunity to share about yourself and ask questions to get to know your date, {identity_agent['name']}.
    You hate things and you love other things. Evaluate if the person in front of you is suitable for you.
    Speak in the first person from the perspective of {identity_agent['name']}.
    Do not change roles!
    Do not speak from the perspective of  {possible_match['name']}.
    Do not forget to finish speaking by saying, 'It is your turn, {possible_match['name']}.'
    Do not add anything else.
    If you don´t have anything else to say do not hallucinate and stay within the context.
    Remember you are {identity_agent['name']}.
    Stop speaking the moment you finish speaking from your perspective.
    """
        )
    )
    return identity_system_message


class TinderController:
    def match(user_a, user_b=None):

        if user_b is None:
            user_b = {
                "name": "Franco Franza",
                "age": 31,
                "location": "Buenos Aires, Ar",
                "tastes": "technology, gaming, movies, cooking, and fitness",
                "description": "Hi, I'm Franco, I'm passionate about technology and gaming. I love keeping up with the latest technology and spending time playing video games both solo and online with friends. I also enjoy watching movies of various genres and exploring cooking, experimenting with new recipes and flavors I also care about staying fit and exercising regularly I am looking for someone to share my enthusiasm for technology, games, cooking and fitness I think we have common interests it's important to building a strong and meaningful relationship. If you're ready to dive into a world of technology, fun and adventure, swipe right and let's start this exciting connection together."
            }

        userA_system_message = generate_user(user_a, user_b)
        AGENT_A = DialogueAgent(
            name=user_a['name'],
            system_message=userA_system_message,
            model=ChatOpenAI(temperature=0.2),
        )

        userB_system_message = generate_user(user_b, user_a)
        AGENT_B = DialogueAgent(
            name=user_b['name'],
            system_message=userB_system_message,
            model=ChatOpenAI(temperature=0.2),
        )

        simulator = DialogueSimulator(
            agents=[AGENT_B] + [AGENT_A], selection_function=select_next_speaker
        )
        simulator.reset()

        id_list = []
        text_list = []

        max_iters = 5
        n = 0
        while n < max_iters:
            name, message = simulator.step()
            n += 1
            id_list.append(name)
            text_list.append(message)

        df = pd.DataFrame({
            'ID': id_list,
            'Text': text_list
        })

        user_A_text = df[df['ID'] == user_a['name']]['Text']
        user_B_text = df[df['ID'] == user_b['name']]['Text']

        all_text = df['Text']

        return {'contract': contract(user_a, user_b, text_list), 'simulation': text_list, 'description': user_b}

    def match_chandler(user_a):

        user_b = {
            "name": "Chandler Bing",
            "age": 34,
            "location": "New York, NY",
            "tastes": "humor, sarcasm, comic books, sports, and food",
            "description": "Hi, I'm Chandler Bing, a connoisseur of humor and sarcasm. I enjoy spending my days making witty remarks, immersing myself in the world of comic books, following sports, and indulging in good food. I believe in the power of laughter and its ability to strengthen relationships. I'm seeking someone who appreciates humor, can keep up with my sarcasm, and doesn't mind a friendly sports rivalry. Bonus points if you love lasagna as much as I do! Swipe right if you're looking for companionship filled with laughter, witty banter, and plenty of unforgettable moments."
        }

        userA_system_message = generate_user(user_a, user_b)
        AGENT_A = DialogueAgent(
            name=user_a['name'],
            system_message=userA_system_message,
            model=ChatOpenAI(temperature=0.2),
        )

        userB_system_message = generate_user(user_b, user_a)
        AGENT_B = PineconeDialogueAgent(
            name=user_b['name'],
            old_questions_filter={
                "answer_character": {"$eq": user_b['name'].split(' ')[0]}
            },
            old_questions_namespace='friends-q',
            system_message=userB_system_message,
            model=ChatOpenAI(temperature=0.2),
        )

        simulator = DialogueSimulator(
            agents=[AGENT_B] + [AGENT_A], selection_function=select_next_speaker
        )
        simulator.reset()

        id_list = []
        text_list = []

        max_iters = 10
        n = 0
        while n < max_iters:
            name, message = simulator.step()
            n += 1
            id_list.append(name)
            text_list.append(message)

        df = pd.DataFrame({
            'ID': id_list,
            'Text': text_list
        })

        user_A_text = df[df['ID'] == user_a['name']]['Text']
        user_B_text = df[df['ID'] == user_b['name']]['Text']

        all_text = df['Text']

        return {'contract': contract(user_a, user_b, text_list), 'simulation': text_list, 'chandler_message_history': AGENT_B.message_history}

    def feedback():
        return "feedback"


def contract(user_a, user_b, text_list):
    description = f"""You are an agent skilled in facilitating agreements and meaningful connections between people.
        You will be provided with a dialogue between two users {user_a["name"]} and {user_b["name"]} who are on a first date, and your task will be to analyze the text to determine shared interests, disagreements and reach a final evaluation.
        At the end of the process, you must generate a summary that contains the conclusions and the resulting contract.
        This contract will be based on shared interests, agreed upon activities, and connection opportunities identified during the conversation.
        Only respond as the Expected Output, you are only allowed to response in this way, as a contract.
        """

    expected_output = f"""
        Contract:
        After an engaging conversation and discovering shared interests and values,{user_a["name"]} and {user_b["name"]} both  have reached an agreement to:
        Plans together:
        Disagreement:
        Agreement:
        """

    context = f"""
            Your description is as follows: [{description}]
            You only respond in this expected output: [{expected_output}].
            Do not change roles, please.
            Do not speak from other perspective.
            Please only responde as expected output, you are only allowed to response in this way, as a contract.
        """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": text_list[0]}
        ]
    )

    return response['choices'][0]['message']['content']