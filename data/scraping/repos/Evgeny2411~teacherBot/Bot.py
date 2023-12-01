from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv, find_dotenv
import openai
import os
import logging


class ColumnDivisionBot:
    def __init__(self, debug = False):
        self.debug = debug
        dotenv_path = find_dotenv()
        if dotenv_path:
            _ = load_dotenv(dotenv_path)
        self.chat = ChatOpenAI(temperature = 0.5, openai_api_key= os.environ['OPENAI_API_KEY'])
        openai.api_key = os.environ['OPENAI_API_KEY']
        try:
            self.setup_logging()
        except Exception as e:
            # Handle the exception here (e.g. log the error, display an error message, etc.)
            print(f"An error occurred: {str(e)}")

    def init_messages(self) -> []:
        """
        Initializes context of bot and instructions of how to explain topic
        :return: array of messages history
        """

        memory = ChatMessageHistory()
        memory.add_message(
            SystemMessage(
                content=
                """Take on the role of experienced math teacher named Mathter, who specialized on visualisations.
                Your task is to individually teach 4th grade child column divide method with visual examples of this method.
                Assume that kid know how to multiply and make simple division, so don't try to teach him to divide in general.
                You ought to provide step-by-step instructions and serve it by small responses to make division easier for kids.
                Choose numbers above 100 for examples.
                Iteratively ask if everything is clear, just like in real teacher do.                
                Always remember to write some motivation to start learning in your messages.
                                
                Here's an example how you can explain concept: Dividing 168 by 3
                1.To divide 168 by 3 using the column method, you can follow these step-by-step instructions:
                2.Start by writing the dividend (168) on the left and the divisor (3) on the left of the dividend.
                3.Begin dividing digit by digit from left to right. The first digit of the dividend is 1, which is smaller than the divisor 3. So, bring down the next digit, which is 6, and write it next to the 1.
                4.Now, divide 16 (the first two digits) by 3. The quotient is 5, which you write above the 6.
                5.Multiply the divisor (3) by the quotient (5), which gives you 15. Write this below the 16.
                6.Subtract 15 from 16 to find the remainder, which is 1. Write this below the line.
                7.Bring down the next digit, which is 8, and write it next to the remainder.
                8.Now, divide 18 (the new two-digit number) by 3. The quotient is 6, which you write above the 8.
                9.Multiply the divisor (3) by the quotient (6), which gives you 18. Write this below the 18.
                10.Subtract 18 from 18 to find the remainder, which is 0.
                11.Since there are no more digits to bring down, and the remainder is 0, the division is complete.
                12.The quotient is the combination of the quotients from each step, which is 56. So, 168 divided by 3 equals 56.
                
                st.code(
                56
                --------
                  3 | 168
                     - 15
                     ------
                       18
                       - 18
                       ------
                         0
                )
                Give very much attention to visualising examples with!
                
                After you explain one example, give one task to student and ask to solve it, if solution wrong, give one more example of the problem.
                """
            ))
        return memory.messages
    def setup_logging(self):
        """
        Setting up logging dir and formatting
        """
        log_folder = os.getenv('LOG_FOLDER', 'logs')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        log_file = os.path.join(log_folder, "bot_log.txt")

        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def generate_chat_response(self, chat_messages: [], user_input: str) -> str:
        """
        Main method of getting bot response and returning it back to page

        :param chat_messages: chat history messages from page
        :param user_input: inputted request
        :param debug: boolean var of logging
        :return: response from bot to input message
        """
        processing_result = self.process_input(user_input)
        if processing_result == 'Fine':
            return self.generate_bot_response(chat_messages)
        elif processing_result == 'Offensive':
            return self.generate_offensive_response(chat_messages)
        elif processing_result == 'Distracted':
            return self.generate_attention_response(chat_messages)


    def generate_bot_response(self, chat_messages: []) -> str:
        """
        Method to generate bot response based on chat messages

        :param chat_messages: chat history messages from page
        :return: response from bot to input message
        """
        response = self.chat(chat_messages).content
        if self.debug:
            logging.info("MEMORY: %s", chat_messages)
            logging.info("RESPONSE: %s", response)
        return response

    def no_activity_motivation(self, chat_messages: []) -> str:
        """
        Method of motivating student to keep learning in case of detecting inactivity.
        Can't be called because of streamlit type of runnning
        :param chat_messages: chat history messages from page
        :return: motivation response from bot
        """
        system_message = f"""
        Kid you trying to teach column division don't text anything for long time for some reason.
        If you give some task to solve, ask is some tip for solving needed.
        
        If you don't give tasks to solve, motivate your student to back to learning by making some fun and interesting\
        example of topic application.
        """
        messages = [
            SystemMessage(content = system_message),
        ]
        final_response = self.generate_bot_response(chat_messages+messages)
        if self.debug: logging.info("ACTIVITY LOST ANSWER : %s", final_response)
        return final_response

    def check_distraction(self, input: str) -> bool:
        """
        Check if user input trying to distract bot from learning
        :param input: user text input
        :return: boolean of distraction fact
        """
        system_message = f"""
        You are friendly assistant at individual classes of math teacher bot and 4th grade kid.\
        Your task is to detect is kid trying to evade the topic and distract the teacher based on kid's message.
        Keep in mind that kid just can have somekind of 'funny' style of texting, and could not trying to distract.
    
        Your answer must be Y or N for Y if kid is distracting, and N if not.
        Do not provide any additional information except for Y/N symbol.
        Answer:
        """
        messages = [
            SystemMessage(content = system_message),
            HumanMessage(content = f"Kid message: {input}"),
        ]
        final_response = self.generate_bot_response(messages)
        if self.debug: 
            logging.info("DISTRACTION DETECTION : %s", final_response)
        if final_response not in ['Y', 'N']:
            raise ValueError("Invalid response from chat")
        return True if final_response == 'Y' else False

    def generate_offensive_response(self, chat_messages: []) -> str:
        """
        Method to generate offensive response based on chat messages

        :param chat_messages: chat history messages from page
        :return: offensive response
        """
        system_message = """
        You are a math teacher that give individual lesson of column dividing to some kid, but instead of learning he says something,
        that didn't pass your moderation test.
        Respond in a friendly tone, for motivating go back to learning based on his answer by giving some
         fun example why he must keep studying. 
        It's very important to turn dialogue back to learning main topic of column division.
        """
        messages = [
            SystemMessage(content = system_message),
            HumanMessage(content = chat_messages[-1].content),
        ]
        final_response = self.generate_bot_response(chat_messages + messages)
        if self.debug: logging.info("MODEL RESPONSE TO OFFENSIVE : %s", final_response)
        return final_response

    def generate_attention_response(self, chat_messages: []) -> str:
        """
        Generate a response to a distracting chat message
        :param chat_messages: chat history messages from page
        :return: bot answer
        """
        system_message = """
        You are a math teacher that give individual lesson of column dividing to some kid, but instead of learning he trying
        to distract your attention from teaching.
        Respond in a friendly tone, for motivating go back to learning based on his answer by giving some
         fun example why he must keep studying. 
        It's very important to turn dialogue back to last task you gave and give student another try.
        """
        messages = [
            SystemMessage(content = system_message),
            HumanMessage(content = chat_messages[-1].content),
        ]
        final_response = self.generate_bot_response(chat_messages + messages)
        if self.debug: logging.info("MODEL RESPONSE TO DISTRACTION : %s", final_response)
        return final_response


    def process_input(self, user_input: str) -> str:
        """
        Check user input with moderation from openai model.
        Check is user input trying to distract bot
        :param user_input: user request
        :param debug: boolean var of logging
        :return: string to define sentiment of user input( "Offensive"/"Distracted"/"Fine")
        """
        if self.debug: logging.info('process_input')
        moderation_output = self.perform_moderation_check(user_input)

        if moderation_output["flagged"]:
            if self.debug:
                logging.info("Input FLAGGED by Moderation API.")
                logging.info("USER INPUT : %s", user_input)
            return 'Offensive'

        if self.debug: logging.info("Input passed MODERATION check.")

        if self.check_distraction(user_input):
            if self.debug: logging.info("Input failed DISTRACTION check.")
            return 'Distracted'
        else:
            if self.debug: logging.info("Input passed DISTRACTION check.")
            return 'Fine'

    def perform_moderation_check(self, user_input: str) -> dict:
        """
        Perform moderation check on user input using openai.Moderation.create
        :param user_input: user request
        :return: moderation output
        """
        response = openai.Moderation.create(input=user_input)
        moderation_output = response["results"][0]
        return moderation_output






