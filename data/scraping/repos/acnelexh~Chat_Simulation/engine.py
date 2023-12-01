# simulation engine
import tqdm
import json
import random
from pathlib import Path
from openai import OpenAI
from persona import PersonaGenerator, generate_emotions_and_arousals

class Engine:
    '''
    Simulation engine for the chatbot
    '''
    def __init__(self, save_dir: str, resume: Path = None):
        # init openai client
        with open("MY_KEY", "r") as f:
            self.client = OpenAI(api_key=f.read().strip())
        # init emotion combination
        self.emotion_combination = generate_emotions_and_arousals() # list of ((emotion1, emotion2), (arousal1, arousal2))
        # check for resume
        if resume is None:
            # init two personas
            self.agents = self._init_agents() # initialize chatbot and user agents
            self.chatbot_persona = str(self.agents[0])
            self.user_persona = str(self.agents[1])
        else:
            # gather left over emotion combination
            # gather persona from resume dir
            self.resume_simulation(resume)
        # init conversation
        self.conversation = [] # list of list of (agent_type, content)
        # file io stuff
        if resume is None:
            self.output_dir = Path(save_dir)
            if self.output_dir.exists() == False:
                self.output_dir.mkdir()
            # save persona to output dir
            with open(f'{self.output_dir}/chatbot_persona.txt', 'w') as f:
                f.write(self.chatbot_persona)
            with open(f'{self.output_dir}/user_persona.txt', 'w') as f:
                f.write(self.user_persona)
        else:
            self.output_dir = resume
        # error io incase of crash from previous simulation
        self.error_log = self.output_dir/"log.txt"
        if self.error_log.exists() == False:
            self.error_log.touch()
    
    def format_save_name(self, emotion_shift):
        '''
        format the save name for the simulation
        args:
            emotion_shift: ((emotion1, emotion2), (arousal1, arousal2))
        return:
            save_name: str, save name for the simulation in the format of
            "emotion1_emotion2_arousal1_arousal2.txt"
        '''
        emotion1, emotion2 = emotion_shift
        #arousal1, arousal2 = emotion_shift[1]
        return f"{emotion1}_{emotion2}.txt"

    def resume_simulation(self, resume_dir: Path):
        '''
        resume simulation from previous simulation
        args:
            resume_dir: Path, path to the resume directory
        '''
        # glob all json file
        json_files = resume_dir.glob("*.json")
        # "emotion1_emotion2_arousal1_arousal2"
        file_stem = [str(f.stem)+".txt"  for f in json_files] 
        # turn them into emotion combination
        left_over = set([self.format_save_name(x) for x in self.emotion_combination]) - set(file_stem)
        if len(left_over) == 0:
            raise ValueError("No simulation to resume")
        # remove emotion combination that has been simulated
        self.emotion_combination = [x for x in self.emotion_combination if self.format_save_name(x) in left_over]
        # read in chatbot and user persona
        self.chatbot_persona = resume_dir.joinpath("chatbot_persona.txt").read_text()
        self.user_persona = resume_dir.joinpath("user_persona.txt").read_text()

    def _init_agents(self) -> str:
        '''
        initialize user persona
        return:
            user_persona: nlp description of the user
        '''
        persona_generator = PersonaGenerator()
        agents = [persona_generator.generate_chatbot(), persona_generator.generate_user()]
        return agents

    def parameter_generation(self, emotion_shift):
        '''
        generate parameter for the current simulation iteration
        args:
            emotion_shift: ((emotion1, emotion2), (arousal1, arousal2))
        return:
            params: dict of parameters
        '''
        emotion1, emotion2 = emotion_shift
        #arousal1, arousal2 = emotion_shift[1]
        emotion_shift = f"({emotion1}) -> ({emotion2})"
        number_of_turns = random.randint(7, 12)
        
        # Define the topics and their corresponding probabilities based on Daily Dialogue Paper
        topics_with_probabilities = {
            "Ordinary Life": 28.26,
            "School Life": 3.69,
            "Culture & Education": 0.42,
            "Attitude & Emotion": 4.95,
            "Relationship": 33.33,
            "Tourism": 8.32,
            "Health": 1.96,
            "Work": 14.49,
            "Politics": 1.00,
            "Finance": 3.59
        }

        # Normalize the probabilities (they should sum to 1)
        total = sum(topics_with_probabilities.values())
        topics_with_normalized_probabilities = {k: v / total for k, v in topics_with_probabilities.items()}

        # Choose a topic based on the distribution
        topics = list(topics_with_normalized_probabilities.keys())
        probabilities = list(topics_with_normalized_probabilities.values())
        topic = random.choices(topics, weights=probabilities, k=1)[0]
        params = {
            'USER STARTING EMOTION': emotion1,
            #'USER STARTING AROUSAL': arousal1,
            'USER ENDING EMOTION': emotion2,
            #'USER ENDING AROUSAL': arousal2,
            'TURNS PER SIMULATION': number_of_turns,
            'TOPIC': topic
        }
        return params

    def prompt_generation(self, chatbot_start, params):
        '''
        generate prompt for the current simulation iteration
        args:
            chatbot_start: bool, whether chatbot start first
            params: dict of parameters
        return:
            msg: str, prompt for the current simulation
        '''
        # randomly select a which agent to start first
        generation_format = "CHATBOT: [...]\nUSER-[EMOTION]: [...]" if chatbot_start else "USER-[EMOTION]: [...]\nCHATBOT: [...]"
        # prompt for message generation
        msg = "Rules for the simulation:\n"
        msg = f"1. Simulate a conversation between the CHATBOT and USER, aligning with their individual persona with the topic {params['TOPIC']}. Begin the conversation skipping formal greetings. This will make the conversation feel more immediate and focused.\n"
        msg += f"2. The USER should only show {params['USER STARTING EMOTION']}, {params['USER ENDING EMOTION']}, and neutral emotion throughout the conversation. USER should start with a initial emotion state of {params['USER STARTING EMOTION']}, through gradual shift in emotion guided by CHATBOT towards the final emotion state of {params['USER ENDING EMOTION']}.\n"
        msg += f"3. The USER’s emotions should shift gradually, not abruptly, to keep the conversation natural. Suggest the chatbot to ask probing questions or make statements that could realistically lead to the final emotion state.\n"
        msg += f"4. Generate {params['TURNS PER SIMULATION']} turns of conversation, with the following format:\n"
        msg += f"{generation_format}\n"
        msg += f"5. Natural Display of Emotion: Use descriptive language that naturally conveys the USER's emotional state through their word choice, tone, and the content of their speech rather than explicitly stating the emotion state.Include subtle cues that indicate a shift in emotion, such as changes in the USER's responsiveness, the length of their messages, or their use of punctuation and capitalization.\n"
        msg += f"6. Detailed and realistic conversation: USER should provide specific details about the trigger of their emotions to make it more believable, e.g. specific relationship drama or dynamic (e.g. cheating husband/wife, missed date, unbalanced relationship dynamic) that contribute to sadness or disgust, specific activity and role models (e.g. reading Socrates, Shakespear, etc) that brings them joy and excitement. \n"
        msg += "7. Adopt the personality described in the character section below and respond to the last message in conversation history. Consider the complete conversation history, the additional context, the character's persona, emotional state and goals below when simulating.\n"
        msg += "8. Avoid Forced Positivity: If the conversation naturally leads to a less positive conclusion, let it be. Not every conversation has to end on a high note, especially if it doesn't fit the flow of the dialogue\n"
        msg += f"9. Varied Conversation Endings: The conversation doesn't need to end with USER thanking the CHATBOT for listening. Allow for a variety of conversation endings that are more aligned with the final emotion state of {params['USER ENDING EMOTION']}.\n"
        msg += """10. Definition of EMOTIONs: 
        Happy/Joy - is often defined as a pleasant emotional state that is characterized by feelings of contentment, joy, gratification, satisfaction, and well-being.
        Sadness - Sadness is another type of emotion often defined as a transient emotional state characterized by feelings of disappointment, grief, hopelessness, disinterest, and dampened mood. Like other emotions, sadness is something that all people experience from time to time. In some cases, people can experience prolonged and severe periods of sadness that can turn into depression. Sadness can be expressed in a number of ways including: Crying, Dampened mood, Lethargy, Quietness, Withdrawal from others.
        Fear - Fear is a powerful emotion that can also play an important role in survival. When you face some sort of danger and experience fear, you go through what is known as the fight or flight response.
        Disgust - This sense of revulsion can originate from a number of things, including an unpleasant taste, sight, or smell. Researchers believe that this emotion evolved as a reaction to foods that might be harmful or fatal. When people smell or taste foods that have gone bad, for example, disgust is a typical reaction. Poor hygiene, infection, blood, rot, and death can also trigger a disgust response. This may be the body's way of avoiding things that may carry transmittable diseases.Digust could also be related to contempt of another person or situation.
        Anger - Anger can be a particularly powerful emotion characterized by feelings of hostility, agitation, frustration, and antagonism towards others. Like fear, anger can play a part in your body's fight or flight response.
        Surprise - Surprise is usually quite brief and is characterized by a physiological startle response following something unexpected. A pleasant surprise would be arriving home to find that your closest friends have gathered to celebrate your birthday. \n"""
        msg += "11. Use daily dialogue examples as reference for the simulation to generate realistic emotion through conversation.\n"
        return msg.strip()

    def simulate(self, emotion_shift):
        '''
        Simulate a entire conversation between chatbot and user agents
        args:
            emotion_shift: ((emotion1, emotion2), (arousal1, arousal2))
        return:
            dialogue: list of (agent_type, content)
        '''
        # randomly select a which agent to start first
        chatbot_start = True if random.random() > 0.5 else False
        params = self.parameter_generation(emotion_shift)
        params_str = "PARAMETER:\n" + ''.join([f'{k}: {v}\n' for k, v in params.items()]).strip()
        # prompt for message generation
        msg = self.prompt_generation(chatbot_start, params)
        system = "CHATBOT PERSONA:\n" + self.chatbot_persona + "\n"
        system += "USER PERSONA:\n" + self.user_persona + "\n"
        dd_example = Path("dd_examples.txt").read_text()
        content = f"{dd_example}\n\n{params_str}\n\n{msg}\n\nResponse:\n"
        # save to tmp to see format, sanity check
        with open(f'{self.output_dir}/{self.format_save_name(emotion_shift)}', 'w') as f:
            f.write("============================================\n")
            f.write("SYSTEM:\n")
            f.write(system)
            f.write("============================================\n")
            f.write("CONTENT:\n")
            f.write(content)
        # send to gpt
        #return None
        response = self.sent_to_gpt(system, content)
        dialogue = self.process_response(response, emotion_shift)
        if len(dialogue) != 0:
            self.conversation.append(dialogue)
        return dialogue

    def process_response(self, response, emotion_shift):
        '''
        process response from chatgpt, assume response is in the right format
        args:
            response: str, response from chatgpt
            emotion_shift: ((emotion1, emotion2), (arousal1, arousal2))
        return:
            dialogue: list of (agent_type, content)
        '''
        new_dialogue = []
        dialogue_parsed = response.strip().split('\n')
        for dialogue in dialogue_parsed:
            if len(dialogue) == 0:
                continue
            try:
                agent, content = dialogue.split(':')
            except ValueError:
                # exist the current simulation and log it
                with open(self.error_log, 'a') as f:
                    f.write('Value Error while simulating: ' + self.format_save_name(emotion_shift) + '\n')
                    f.write('Response: ' + response + '\n')
                    return []
            if agent.lower() == 'chatbot':
                new_dialogue.append((1, content))
            else:
                try: # USER-[EMOTION]
                    agent, emotion = agent.split('-')
                    new_dialogue.append((2, content, emotion.strip('[').strip(']')))
                except ValueError:
                    # exist the current simulation and log it
                    with open(self.error_log, 'a') as f:
                        f.write('Value Error while simulating: ' + self.format_save_name(emotion_shift) + '\n')
                        f.write('Response: ' + response + '\n')
                        return []
        return new_dialogue        

    def start(self):
        '''
        start simulation
        '''
        # end conversation after 100 turn
        # simulate over all possible emotion combination
        for emotion_shift in tqdm.tqdm(self.emotion_combination):
            dialogue = self.simulate(emotion_shift)
            if len(dialogue) == 0:
                # dont save if error, just log it and resume to the next simulation
                continue
            # save to output dir
            with open(f'{self.output_dir}/{self.format_save_name(emotion_shift)[:-4]}.json', 'w') as f:
                f.write(json.dumps(dialogue))
    
    def sent_to_gpt(self, system: int, content: str):
        '''
        send message to gpt for response
        args:
            system: str, system persona
            content: str, content of the message
        return:
            
        '''
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content}])
        # save to output dir and parse response
        response = completion.choices[0].message.content
        return response
