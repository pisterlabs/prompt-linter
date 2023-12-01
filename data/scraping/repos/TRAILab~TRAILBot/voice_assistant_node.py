import os
import re
import time

import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from trailbot_interfaces.msg import SnacksInventory
from trailbot_interfaces.srv import SnackWanted

from .conversation_handler import ConversationHandler
from .speech_recognizer import SpeechRecognizer
from .text_to_speech_engine import ElevenLabsEngine, Pyttsx3Engine
from .show_emojis import Emojis


class VoiceAssistant(Node):
    def __init__(self):
        """
        Listens and comprehends user's commands,
        Sends snack_wanted request to behaviour planner,
        Chats with user using chatGPT

        """
        super().__init__('voice_assistant_node')

        # Emojis gui
        self.gui = Emojis(self.get_logger())

        # Declare params
        self.declare_parameter('exit_cmd_options', [
                               'bye', 'bubye', 'adios', 'ciao', 'thanks', 'thank you'])
        self.declare_parameter('speech_recognizer.mic_device_index', 12)
        self.declare_parameter('speech_recognizer.energy_threshold', 200)
        self.declare_parameter('speech_recognizer.timeout', 4)
        self.declare_parameter('speech_recognizer.phrase_time_limit', 4)
        # if True, the node uses openai transcriber otherwise it uses google transcriber.
        self.declare_parameter('speech_recognizer.use_whisper', True)
        # options: pyttsx3, elevenlabs
        self.declare_parameter('text_to_speech_engine', 'elevenlabs')

        # Get params
        self.exit_cmd_options = self.get_parameter('exit_cmd_options').value
        self.snack_options = []
        self.snack_quantity = []

        # Print params
        use_whisper = self.get_parameter('speech_recognizer.use_whisper').value
        self.get_logger().info(
            f'use_whisper: {use_whisper}, exit_cmd_options: {self.exit_cmd_options}')

        # openAI set-up
        personality = "You are a helpful assistant."
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        self.messages = [{"role": "system", "content": f"{personality}"}]

        # Text to Speech Generator set-up
        text_to_speech_engine = self.get_parameter(
            'text_to_speech_engine').value
        if text_to_speech_engine == 'pyttsx3':
            self.tts_engine = Pyttsx3Engine()
        elif text_to_speech_engine == 'elevenlabs':
            self.tts_engine = ElevenLabsEngine()
        else:
            raise NotImplementedError

        # Speech Recognizer set-up
        # Note: You can list microphones with this command and set the device_index
        # to the index of your favourite microphone from the list
        self.get_logger().info('Available mics:')
        mic_list = SpeechRecognizer.list_mics()
        # dictionary with microphones and their indexes
        for index, key in enumerate(mic_list):
            self.get_logger().info(f'{key}: {index}')
        self.get_logger().info('Mic index in use: %s' % str(
            self.get_parameter('speech_recognizer.mic_device_index').value))
        self.get_logger().info('speech_recognizer.energy_threshold: %s' %
                               str(self.get_parameter('speech_recognizer.energy_threshold').value))
        self.speech_recognizer = SpeechRecognizer(mic_index=self.get_parameter('speech_recognizer.mic_device_index').value,
                                                  energy_threshold=self.get_parameter(
                                                      'speech_recognizer.energy_threshold').value,
                                                  use_whisper=self.get_parameter(
                                                      'speech_recognizer.use_whisper').value,
                                                  timeout=self.get_parameter(
                                                      'speech_recognizer.timeout').value,
                                                  phrase_time_limit=self.get_parameter(
                                                      'speech_recognizer.phrase_time_limit').value,
                                                  logger=self.get_logger(),
                                                  gui=self.gui)

        # Set-up conversation_handler to save conversations
        self.conversation_handler = ConversationHandler(
            self.messages, save_foldername='conversations')

        # Set up SnackWanted Service client
        # Initialize snack_wanted client and request
        self.cli = self.create_client(SnackWanted, 'snack_wanted')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('snack_wanted service not available, waiting again...')
        self.snack_wanted_request = SnackWanted.Request()
        # if user has said bye (or one of the self.exit_cmd_options)
        self.end_chat = False
        self.in_query_state = False  # True if robot's state is 'QueryState', False otherwise

        # Subscriber: to detect the state of the robot
        self.state_subscriber = self.create_subscription(
            String,
            'trailbot_state',
            self.state_listener_callback,
            5)
        self.state_subscriber  # To prevent unused variable warning

        # Publisher: to let behaviour planner know that the user has ended the chat
        self.publisher = self.create_publisher(Bool, 'query_complete', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(
            timer_period, self.publisher_timer_callback)

        # Subscriber: receive the current snack inventory
        self.snacks_inventory_subscriber = self.create_subscription(
            SnacksInventory,
            'snack_inventory',
            self.available_snacks_listener_callback,
            10)
        self.snacks_inventory_subscriber  # prevent unused variable warning
        self.state = 'None'

    def available_snacks_listener_callback(self, msg):
        self.snack_options = msg.snacks
        self.snack_quantity = msg.quantity

    def publisher_timer_callback(self):
        msg = Bool()
        msg.data = self.end_chat
        self.publisher.publish(msg)

    def state_listener_callback(self, msg):
        # Activate chatbot if in 'QueryState'
        if msg.data == 'QueryState' and self.state != 'QueryState':
            self.state = msg.data
            self.in_query_state = True
            self.end_chat = False
            self.get_logger().info('State changed to: "%s"' % msg.data)
            self.get_logger().info('Activating chatbot!')
        else:
            self.in_query_state = False

    def get_available_snacks(self):
        rclpy.spin_once(self)
        available_snack_options = [
            self.snack_options[i] for i, quantity in enumerate(self.snack_quantity) if quantity > 0]
        return available_snack_options

    def get_available_snacks_str(self):
        available_snack_options = self.get_available_snacks()
        available_snack_options[-1] = 'and ' + available_snack_options[-1]
        available_snack_options_str = " ".join(available_snack_options)
        # print('Available snacks: ', available_snack_options_str)
        return available_snack_options_str

    def send_request(self, snack_wanted):
        """
        Sends request to snack_wanted service 

        Args:
            snack (str): snack requested by user e.g. 'chips'

        Returns:
            response (bool, str): response of snack_wanted service 
                                    e.g. response.success = 'True', 
                                    response.message = 'Snack successfully dispensed'
        """

        self.snack_wanted_request.snack = snack_wanted
        self.future = self.cli.call_async(self.snack_wanted_request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def request_snacks(self, snack_wanted):
        self.speak(f'Alrighty, {snack_wanted} coming right up!')

        # Send SnackWanted request to behaviour planner
        response = self.send_request(snack_wanted)
        self.get_logger().info(
            f'Response of snack_wanted service request: {response.success}, {response.message}')

        if response.success:
            self.speak(f'Please pick up your {snack_wanted}.')
        else:
            self.speak(f'{response.message}')

        return response.success

    def speak(self, msg_list):
        """ Text to speech generation

        Args:
            msg_list (list of str): list of sentences to speak
        """
        self.gui.show_speaking()
        self.tts_engine.speak(msg_list)

    def look_for_keywords(self, user_input, keywords):
        """ Look for keywords in user's prompt
        Upon detection return True and the word found

        Args:
            user_input (str): transcribed voice command
            keywords (list of str): list of keywords to look for
        Return:
            bool: if any keyword is found in user_input
            word (str): keyword found
        """

        # Make capital letters to lower and remove punctuation
        user_input = re.sub(r'[^\w\s]', '', user_input.lower())

        for word in keywords:
            if word in user_input:
                self.get_logger().info('Word found: "%s"' % word)
                return True, word

        return False, None

    def say_bye(self):
        self.speak(f'Nice chatting with you. Have a nice day!')
        self.end_chat = True

    def find_bye(self, user_input):
        end_chat, _ = self.look_for_keywords(user_input, self.exit_cmd_options)
        if end_chat:
            return True
        return False

    def find_snack_in_input(self, user_input):
        # Find the requested snack in user_input
        want_snacks, snack_wanted = self.look_for_keywords(
            user_input, self.snack_options)
        return want_snacks, snack_wanted

    def chat_with_user(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        # Print available openai models with:
        # print(openai.Model.list())
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0.8
        )

        response = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        # print(f"\n{response}\n")
        self.conversation_handler.save_inprogress(self.messages)

        self.speak(f'{response}')

    def process_user_input(self, user_input):
        # Keep chatting until user goes silent
        while rclpy.ok() and user_input is not None:

            if self.find_bye(user_input):
                self.say_bye()
                break

            want_snacks, snack_wanted = self.find_snack_in_input(user_input)

            if want_snacks:
                # Request snacks from behaviour planner
                success = self.request_snacks(snack_wanted)
                if success:
                    self.speak(f'Would you like anything else?')
                else:
                    self.speak(
                        f'We have {self.get_available_snacks_str()}. What would you like?')
            else:
                self.chat_with_user(user_input)

            # Get user prompt
            user_input = self.speech_recognizer.get_input()

        # If user_input is None i.e user is silent, say bye
        if user_input is None:
            self.say_bye()

    def run(self):
        if self.in_query_state:

            # Introduce upon reaching the human
            available_snacks = self.get_available_snacks_str()
            intro = f'Hi! Would you like to have snacks? I have {available_snacks}. What would you like?'
            self.speak(intro)

            # Get user prompt
            user_input = self.speech_recognizer.get_input()

            # End conversation if user did not say anything
            if user_input is None:
                self.say_bye()
            else:
                self.process_user_input(user_input)


def main(args=None):
    rclpy.init(args=args)

    voice_assistant_node = VoiceAssistant()

    while rclpy.ok():
        rclpy.spin_once(voice_assistant_node)
        # # Allow time for message to be published on /query_complete topic and state to be changed
        # time.sleep(1)
        # rclpy.spin_once(voice_assistant_node)
        voice_assistant_node.run()

    voice_assistant_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
